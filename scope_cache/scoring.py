"""Показатели гранат и игроков; наблюдения отделены от прогнозов."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from scope_cache import ROOT
from scope_cache.core import save_analysis, sha256_file
from scope_cache.data import BLIND_COUNTS, RELEASE_FEATURES, event_record
from scope_cache.geometry import MapGeometry, SMOKE_PARAMETERS
from scope_cache.model import SETUP_FEATURES, load, matrix, predict, predict_setup, probabilities


def read_rows(path):
    path = Path(path)
    with path.open(encoding="utf-8-sig", newline="") as stream:
        if path.suffix.lower() == ".csv":
            rows = list(csv.DictReader(stream))
            numeric = set(SETUP_FEATURES + RELEASE_FEATURES + BLIND_COUNTS)
            numeric.update(["impact", "end_x", "end_y", "end_z", "fire_area_xy"])
            numeric.update(
                f"{side}_{value}"
                for side in ["enemy", "ally", "self"]
                for value in ["damage", "flash_added"]
            )
            for row in rows:
                for key in numeric.intersection(row):
                    row[key] = float(row[key]) if row[key].strip() else None
        else:
            rows = json.load(stream)
            rows = rows if isinstance(rows, list) else [rows]
    if not rows or not all(isinstance(row, dict) for row in rows):
        raise ValueError("Нужен объект гранаты или непустой список объектов")
    return rows


def outcome_status(kind, impact):
    if kind == "Smokes":
        return "not_applicable_smoke"
    if impact is None:
        return "unknown"
    return "known_positive" if impact else "known_zero"


def observed_columns(events):
    types = events["type"].to_numpy()
    known = np.isfinite(events["impact"].to_numpy())
    columns = {
        "observed_impact": events["impact"],
        "outcome_status": pa.array(
            [
                outcome_status(kind, impact)
                for kind, impact in zip(types, events["impact"].to_pylist())
            ]
        ),
    }
    for key in [
        "enemy_damage",
        "ally_damage",
        "self_damage",
        "enemy_flash_added",
        "ally_flash_added",
        "self_flash_added",
    ]:
        applies = types == "Flashbangs" if "flash" in key else np.isin(types, ["HEs", "Fires"])
        columns["observed_" + key] = pa.array(events[key].to_numpy(), mask=~(known & applies))
    return columns


def score(row, geometry=None, flash=None, mode="auto"):
    if row.get("map_name", "de_cache") != "de_cache":
        raise ValueError("Модель и геометрия поддерживают только de_cache")
    if mode not in {"auto", "setup", "release"}:
        raise ValueError("Режим прогноза: auto, setup или release")
    kind = row.get("type")
    if kind not in {"Smokes", "Flashbangs", "HEs", "Fires"}:
        raise ValueError("Тип гранаты: Smokes, Flashbangs, HEs или Fires")
    if "GeneralInfo" in row:
        original = row
        row = event_record(row.get("match_id", "inference"), row, kind)
        row.update({k: original[k] for k in RELEASE_FEATURES if k in original})
    known = row.get("impact") is not None
    if known and row["impact"] not in (0, 1):
        raise ValueError("impact: 0, 1 или null")
    result = {
        "type": kind,
        "observed_impact": row.get("impact") if kind != "Smokes" else None,
        "outcome_status": outcome_status(kind, row.get("impact")),
        "tactical_success": None,
    }
    if kind == "Smokes":
        if any(row.get("end_" + a) is None for a in "xyz"):
            raise ValueError("Для смока нужны end_x, end_y, end_z — точка детонации")
        geometry = geometry or MapGeometry()
        result["smoke"] = geometry.smoke_at([row["end_" + a] for a in "xyz"], quantize=False)
        result["smoke"]["interpretation"] = (
            "Геометрическое перекрытие линий обзора; фактический обзор противника неизвестен"
        )
    elif kind == "Flashbangs":
        flash = flash or load()
        missing = [k for k in RELEASE_FEATURES if row.get(k) is None]
        if mode == "release" and missing:
            raise ValueError("Не хватает контекста броска: " + ", ".join(missing))
        use_release = mode == "release" or (mode == "auto" and not missing)
        result["prediction"] = (predict if use_release else predict_setup)([row], *flash)[0]
        if missing:
            result["prediction"]["missing_release_features"] = missing
        for key in [
            "enemy_flash_added",
            "ally_flash_added",
            "self_flash_added",
            "enemy_blinded_players",
        ]:
            result["observed_" + key] = row.get(key) if known else None
    else:
        for key in ["enemy_damage", "ally_damage", "self_damage"]:
            result["observed_" + key] = row.get(key) if known else None
        result["unit"] = "HP"
        if kind == "Fires":
            result["recorded_fire_area_xy"] = row.get("fire_area_xy")
    return result


def score_dataset(root=ROOT, force=False):
    root = Path(root)
    path = root / "data/grenade_scores.parquet"
    events_path, context_path = root / "data/events.parquet", root / "data/release_context.parquet"
    bundle, meta = load(root / "models/flash.joblib")
    if (
        sha256_file(events_path) != meta["source"]["events_sha256"]
        or sha256_file(context_path) != meta["source"]["context_sha256"]
    ):
        raise ValueError("Контрольные суммы данных не соответствуют обучающей выборке")
    metadata_path = path.with_suffix(".metadata.json")
    if not force and path.exists() and metadata_path.exists():
        cached = json.loads(metadata_path.read_text())
        if (
            cached.get("schema_version") == 4
            and cached.get("model_sha256") == meta["artifact"]["sha256"]
            and cached.get("events_sha256") == meta["source"]["events_sha256"]
            and cached.get("context_sha256") == meta["source"]["context_sha256"]
            and cached.get("map_manifest_sha256") == sha256_file(root / "map_assets/manifest.json")
            and cached.get("scores_sha256") == sha256_file(path)
        ):
            return cached
    geometry = MapGeometry(root / "map_assets")
    saved = pq.read_table(root / "models/flash.validation.parquet")
    batches = pq.ParquetFile(events_path).iter_batches(batch_size=25000)
    contexts = pq.ParquetFile(context_path).iter_batches(batch_size=25000)
    temporary = path.with_suffix(".building.parquet")
    writer, offset, known_offset, smoke_cache = None, 0, 0, {}
    try:
        for batch, context_batch in zip(batches, contexts, strict=True):
            events, context = pa.Table.from_batches([batch]), pa.Table.from_batches([context_batch])
            n = len(events)
            if len(context) != n or not np.array_equal(
                context["source_row_index"].to_numpy(), np.arange(offset, offset + n)
            ):
                raise ValueError("Нарушено соответствие строк контекста броскам")
            types = events["type"].to_numpy()
            known = np.isfinite(events["impact"].to_numpy())
            columns = {"source_row_index": pa.array(np.arange(offset, offset + n))}
            for key in ["match_id", "player_id", "round_index", "projectile_id", "side", "type"]:
                columns[key] = events[key]
            columns.update(observed_columns(events))
            columns["recorded_fire_area_xy"] = events["fire_area_xy"]
            for key in BLIND_COUNTS:
                columns["observed_" + key] = context[key]
            p, amount, count = [np.full(n, np.nan) for _ in range(3)]
            origin = np.full(n, None, dtype=object)
            indices = np.flatnonzero(types == "Flashbangs")
            if len(indices):
                table = events.take(pa.array(indices))
                for key in RELEASE_FEATURES:
                    table = table.append_column(key, context[key].take(pa.array(indices)))
                pp, aa, cc = probabilities(bundle, matrix(table))
                p[indices], amount[indices], count[indices] = pp, aa, cc
                origin[indices] = "unlabeled_deployment_all_known"
                known_indices = indices[known[indices]]
                records = saved.slice(known_offset, len(known_indices))
                known_offset += len(known_indices)
                if not np.array_equal(
                    records["source_row_index"].to_numpy(), offset + known_indices
                ):
                    raise ValueError("Нарушен порядок проверочных прогнозов")
                p[known_indices] = records["probability"].to_numpy()
                amount[known_indices] = records["conditional_flash_added"].to_numpy()
                count[known_indices] = records["conditional_blinded_players"].to_numpy()
                origin[known_indices] = np.where(
                    records["partition"].to_numpy() == "test", "heldout_match", "train_out_of_fold"
                )
            for key, values in [
                ("predicted_flash_probability", p),
                ("expected_registered_flash_added", p * amount),
                ("expected_blinded_enemy_players", p * count),
            ]:
                columns[key] = pa.array(values, mask=~np.isfinite(values))
            columns["flash_prediction_source"] = pa.array(origin, type=pa.string())
            threshold = meta["high_precision_threshold"]
            columns["flash_high_precision_positive"] = pa.array(
                [
                    bool(v >= threshold) if np.isfinite(v) and threshold is not None else None
                    for v in p
                ],
                type=pa.bool_(),
            )
            coverage = {
                key: np.full(n, np.nan)
                for key in [
                    "potential_smoke_coverage",
                    "smoke_coverage_radius96",
                    "smoke_coverage_radius192",
                ]
            }
            blocked, candidates = np.zeros(n, dtype=np.int64), np.zeros(n, dtype=np.int64)
            smoke_indices = np.flatnonzero(types == "Smokes")
            endpoints = np.column_stack([events["end_" + a].to_numpy() for a in "xyz"])
            for i in smoke_indices:
                key = tuple(
                    np.round(endpoints[i] / SMOKE_PARAMETERS["quantization_units"]).astype(int)
                )
                if key not in smoke_cache:
                    smoke_cache[key] = geometry.smoke_at(endpoints[i])
                value = smoke_cache[key]
                for destination, source in [
                    ("potential_smoke_coverage", "potential_coverage"),
                    ("smoke_coverage_radius96", "coverage_radius_96"),
                    ("smoke_coverage_radius192", "coverage_radius_192"),
                ]:
                    coverage[destination][i] = (
                        value[source] if value[source] is not None else np.nan
                    )
                blocked[i], candidates[i] = (
                    value["blocked_static_sightlines"],
                    value["candidate_static_sightlines"],
                )
            for key, values in coverage.items():
                columns[key] = pa.array(values, mask=~np.isfinite(values))
            for key, values in [
                ("blocked_static_sightlines", blocked),
                ("candidate_static_sightlines", candidates),
            ]:
                columns[key] = pa.array(values, mask=types != "Smokes")
            columns["actual_enemy_view_blocked"] = pa.nulls(n, type=pa.bool_())
            table = pa.table(columns)
            if writer is None:
                writer = pq.ParquetWriter(
                    temporary,
                    table.schema,
                    compression="zstd",
                    compression_level=9,
                    use_dictionary=[
                        "match_id",
                        "player_id",
                        "type",
                        "outcome_status",
                        "flash_prediction_source",
                    ],
                    use_byte_stream_split=[
                        f.name for f in table.schema if pa.types.is_floating(f.type)
                    ],
                )
            writer.write_table(table)
            offset += n
            if offset % 250000 == 0:
                print(f"Оценено {offset:,} бросков", flush=True)
        if known_offset != len(saved):
            raise ValueError("Обработаны не все известные исходы")
        writer.close()
        temporary.replace(path)
    except BaseException:
        if writer is not None:
            writer.close()
        temporary.unlink(missing_ok=True)
        raise
    report = {
        "schema_version": 4,
        "rows": offset,
        "events_sha256": meta["source"]["events_sha256"],
        "context_sha256": meta["source"]["context_sha256"],
        "model_sha256": meta["artifact"]["sha256"],
        "map_manifest_sha256": sha256_file(root / "map_assets/manifest.json"),
        "scores_sha256": sha256_file(path),
        "flash_model": meta["selected_model"],
        "flash_prediction_sources": "Known outcomes: match OOF / held-out; unknown outcomes: separate deployment predictions",
        "smoke_interpretation": "Approximate static line-of-sight coverage; actual enemy vision is unknown",
    }
    metadata_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Показатели рассчитаны: {offset:,} бросков")
    return report


def player_profiles(root=ROOT):
    root = Path(root)
    cols = [
        "match_id",
        "player_id",
        "type",
        "observed_impact",
        "observed_enemy_damage",
        "observed_ally_damage",
        "observed_self_damage",
        "observed_enemy_flash_added",
        "observed_ally_flash_added",
        "observed_self_flash_added",
        "potential_smoke_coverage",
        "predicted_flash_probability",
        "observed_enemy_blinded_players",
        "observed_ally_blinded_players",
    ]
    frame = pq.read_table(root / "data/grenade_scores.parquet", columns=cols).to_pandas()
    grouped = frame.groupby("player_id", sort=True)
    profiles = grouped.agg(matches=("match_id", "nunique"), total_throws=("type", "size"))
    for kind, prefix in [
        ("Flashbangs", "flash"),
        ("HEs", "he"),
        ("Fires", "fire"),
        ("Smokes", "smoke"),
    ]:
        part = frame[frame.type == kind].copy()
        g = part.groupby("player_id", sort=True)
        profiles[prefix + "_throws"] = g.size()
        profiles[prefix + "_known"] = g.observed_impact.count()
        if kind != "Smokes":
            profiles[prefix + "_registered_positive"] = g.observed_impact.sum(min_count=1)
            profiles[prefix + "_known_effect_rate"] = (
                profiles[prefix + "_registered_positive"] / profiles[prefix + "_known"]
            )
            profiles[prefix + "_coverage"] = (
                profiles[prefix + "_known"] / profiles[prefix + "_throws"]
            )
        else:
            profiles["smoke_mean_static_potential"] = g.potential_smoke_coverage.mean()
            profiles["smoke_geometric_coverage"] = g.potential_smoke_coverage.count() / g.size()
        for field in (
            ["observed_enemy_damage", "observed_ally_damage", "observed_self_damage"]
            if kind in ["HEs", "Fires"]
            else [
                "observed_enemy_flash_added",
                "observed_ally_flash_added",
                "observed_self_flash_added",
                "observed_enemy_blinded_players",
                "observed_ally_blinded_players",
            ]
            if kind == "Flashbangs"
            else []
        ):
            profiles[prefix + "_" + field.removeprefix("observed_")] = g[field].sum(min_count=1)
    flashes = frame[(frame.type == "Flashbangs") & frame.observed_impact.notna()].copy()
    flashes["above_expectation"] = flashes.observed_impact - flashes.predicted_flash_probability
    by_match = flashes.groupby(["player_id", "match_id"]).agg(
        residual=("above_expectation", "sum"), known=("above_expectation", "size")
    )
    profiles["flash_above_expectation"] = flashes.groupby("player_id").above_expectation.mean()
    profiles["flash_matches"] = flashes.groupby("player_id").match_id.nunique()
    profiles["flash_largest_match_share"] = (
        by_match.groupby("player_id").known.max() / flashes.groupby("player_id").size()
    )
    eligible = (
        (profiles.flash_known >= 50)
        & (profiles.flash_matches >= 5)
        & (profiles.flash_largest_match_share <= 0.5)
    )
    profiles["flash_estimation_eligible"] = eligible
    profiles["flash_above_expectation_ci_low"] = np.nan
    profiles["flash_above_expectation_ci_high"] = np.nan
    rng = np.random.default_rng(819)
    for pid in profiles.index[eligible]:
        part = by_match.loc[pid]
        a = part.residual.to_numpy()
        n = part.known.to_numpy()
        take = rng.integers(0, len(part), size=(300, len(part)))
        estimates = a[take].sum(1) / n[take].sum(1)
        low, high = np.quantile(estimates, [0.025, 0.975])
        profiles.loc[pid, ["flash_above_expectation_ci_low", "flash_above_expectation_ci_high"]] = [
            low,
            high,
        ]
    # Only counts are filled with zero. Missing measured amounts/rates remain null.
    counts = [c for c in profiles.columns if c.endswith(("_throws", "_known"))]
    profiles[counts] = profiles[counts].fillna(0).astype("int64")
    profiles = profiles.reset_index()
    pq.write_table(
        pa.Table.from_pandas(profiles, preserve_index=False),
        root / "data/player_utility_profiles.parquet",
        compression="zstd",
    )
    report = {
        "players_with_recorded_throws": len(profiles),
        "eligible_flash_estimates": int(eligible.sum()),
        "eligibility": "At least 50 known flashes, five matches, and no match contributing over 50% of the known flashes",
        "units": "HP, distinct blinded-player events, native FlashDurationAdded, static line ratio; these are never added into one score",
        "flash_above_expectation": "Mean(observed binary flash effect minus match-held-out model probability), known outcomes only; retrospective model-relative residual, not causal contribution or an overall player rating",
        "interval": "300 whole-match bootstrap resamples for eligible players; null for sparse records",
        "coverage": "Known/total outcomes shown separately by grenade type; amounts sum recorded known effects only",
        "limitations": [
            "Unobserved opponent context and missing outcomes can bias player comparisons",
            "The same player can appear in multiple source matches; no prospective skill ranking",
            "No KPD, wins or survival is fabricated from grenade records",
        ],
    }
    save_analysis("players", report, root)
    print(json.dumps(report, indent=2), flush=True)
    return report
