"""Проверка модели, исходных данных и геометрии смока."""

from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from awpy import NavMesh
from scipy.spatial import cKDTree
from scipy.stats import spearmanr
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits
from scope_cache import ROOT, ASSETS
from scope_cache.core import metrics, sha256_file, split_indices, read_analysis, save_analysis
from scope_cache.data import BLIND_COUNTS, RELEASE_FEATURES
from scope_cache.geometry import MapGeometry, SMOKE_PARAMETERS, verify_assets, read_mesh
from scope_cache.model import (
    DEFAULT_MODEL,
    FEATURES,
    SETUP_FEATURES,
    balanced_threshold,
    fingerprint,
    flash_table,
    load,
    matrix,
    precision_threshold,
    probabilities,
    selective_metrics,
)


class WeightedMetrics:

    def __init__(self, y, p, threshold=0.5):
        self.y = np.asarray(y, float)
        self.p = np.asarray(p, float)
        self.pred = self.p >= threshold
        self.order = np.argsort(-self.p, kind="stable")
        values = self.p[self.order]
        self.ends = np.flatnonzero(np.r_[values[:-1] != values[1:], True])

    def evaluate(self, weight):
        weight = np.asarray(weight, float)
        y, p, pred = self.y, self.p, self.pred
        total = weight.sum()
        positive = np.dot(weight, y)
        negative = total - positive
        if not total or not positive or not negative:
            return None
        sorted_w = weight[self.order]
        sorted_y = y[self.order]
        tp = np.cumsum(sorted_w * sorted_y)[self.ends]
        fp = np.cumsum(sorted_w * (1 - sorted_y))[self.ends]
        tpr = np.r_[0, tp / positive]
        fpr = np.r_[0, fp / negative]
        precision = tp / np.maximum(tp + fp, 1e-30)
        recall = tp / positive
        true_pos = np.dot(weight, y * pred)
        true_neg = np.dot(weight, (1 - y) * (~pred))
        clipped = np.clip(p, np.finfo(float).eps, 1 - np.finfo(float).eps)
        return {
            "roc_auc": float(np.trapezoid(tpr, fpr)),
            "average_precision": float(np.sum(np.diff(np.r_[0, recall]) * precision)),
            "accuracy": float((true_pos + true_neg) / total),
            "balanced_accuracy": float(0.5 * (true_pos / positive + true_neg / negative)),
            "brier": float(np.dot(weight, (p - y) ** 2) / total),
            "log_loss": float(
                -np.dot(weight, y * np.log(clipped) + (1 - y) * np.log1p(-clipped)) / total
            ),
        }


def cluster_intervals(y, p, groups, threshold=0.5, baseline=None, repetitions=300, seed=714):
    ids, inverse = np.unique(groups, return_inverse=True)
    scorer = WeightedMetrics(y, p, threshold)
    other = WeightedMetrics(y, baseline, threshold) if baseline is not None else None
    reference = scorer.evaluate(np.ones(len(y)))
    values = {key: [] for key in reference}
    differences = {key: [] for key in reference}
    rng = np.random.default_rng(seed)
    for _ in range(repetitions):
        frequency = np.bincount(rng.integers(0, len(ids), len(ids)), minlength=len(ids))
        weight = frequency[inverse]
        score = scorer.evaluate(weight)
        if score is None:
            continue
        comparison = other.evaluate(weight) if other is not None else None
        for key in values:
            values[key].append(score[key])
            if comparison is not None:
                differences[key].append(score[key] - comparison[key])
    result = {
        "method": "Percentile bootstrap of whole match clusters, event-weighted metrics",
        "clusters": len(ids),
        "repetitions": repetitions,
        "seed": seed,
        "confidence_level": 0.95,
        "metrics": {
            key: {
                "estimate": reference[key],
                "ci95": np.quantile(v, [0.025, 0.975]).tolist(),
            }
            for key, v in values.items()
        },
    }
    if other is not None:
        base = other.evaluate(np.ones(len(y)))
        result["paired_difference_vs_setup"] = {
            key: {
                "estimate": reference[key] - base[key],
                "ci95": np.quantile(v, [0.025, 0.975]).tolist(),
            }
            for key, v in differences.items()
        }
    return result


def selective_interval(y, p, groups, threshold, repetitions=300):
    result = selective_metrics(y, p, threshold)
    if threshold is None:
        return result
    ids, inverse = np.unique(groups, return_inverse=True)
    chosen = p >= threshold
    correct = np.bincount(inverse, weights=np.asarray(y) * chosen, minlength=len(ids))
    n = np.bincount(inverse, weights=chosen, minlength=len(ids))
    rng = np.random.default_rng(715)
    estimates = []
    for _ in range(repetitions):
        take = rng.integers(0, len(ids), len(ids))
        denominator = n[take].sum()
        if denominator:
            estimates.append(float(correct[take].sum() / denominator))
    result["precision_ci95"] = (
        np.quantile(estimates, [0.025, 0.975]).tolist() if estimates else None
    )
    result["method"] = (
        "Percentile bootstrap of whole held-out matches; threshold frozen using training OOF only"
    )
    return result


def diagnose_model(model_path=DEFAULT_MODEL, root=ROOT):
    root = Path(root)
    model_path = Path(model_path)
    meta = json.loads(model_path.with_suffix(".metadata.json").read_text())
    table = flash_table(root / "data" / "events.parquet", root / "data" / "release_context.parquet")
    x = matrix(table)
    y = table["impact"].to_numpy().astype(np.int8)
    groups = table["match_id"].to_numpy()
    records = pq.read_table(model_path.with_suffix(".validation.parquet"))
    test = records["partition"].to_numpy() == "test"
    p = records["probability"].to_numpy()
    uncertainty = cluster_intervals(
        y[test],
        p[test],
        groups[test],
        meta["decision_threshold"],
        records["setup_probability"].to_numpy()[test],
    )
    selective = selective_interval(y[test], p[test], groups[test], meta["high_precision_threshold"])
    slices = []
    train_players = set(table["player_id"].to_numpy()[~test])
    players = table["player_id"].to_numpy()
    conditions = {
        "T": table["side"].to_numpy() == 2,
        "CT": table["side"].to_numpy() == 3,
        "round_0_5": table["round_index"].to_numpy() < 6,
        "round_6_11": (table["round_index"].to_numpy() >= 6)
        & (table["round_index"].to_numpy() < 12),
        "round_12_plus": table["round_index"].to_numpy() >= 12,
        "airborne": table["is_airborne"].to_numpy() == 1,
        "grounded": table["is_airborne"].to_numpy() == 0,
        "unseen_thrower": np.array([pid not in train_players for pid in players]),
        "seen_thrower": np.array([pid in train_players for pid in players]),
    }
    for name, mask in conditions.items():
        choose = mask & test
        if choose.any():
            slices.append(
                {
                    "slice": name,
                    "rows": int(choose.sum()),
                    "matches": len(set(groups[choose])),
                    "metrics": metrics(y[choose], p[choose], meta["decision_threshold"]),
                }
            )
    tr, te, cv, _ = split_indices(x, y, groups, 42, meta["selection"]["fold_count"])
    prototype = HistGradientBoostingClassifier(
        max_iter=200,
        max_leaf_nodes=31,
        min_samples_leaf=100,
        l2_regularization=10,
        early_stopping=False,
        random_state=42,
    )
    setup_oof = np.full(len(tr), np.nan)
    with threadpool_limits(limits=2):
        for i, (a, b) in enumerate(cv):
            fitted = clone(prototype).fit(x[tr[a], : len(SETUP_FEATURES)], y[tr[a]])
            setup_oof[b] = fitted.predict_proba(x[tr[b], : len(SETUP_FEATURES)])[:, 1]
            print("Matched-capacity setup ablation", i + 1, flush=True)
        fitted = clone(prototype).fit(x[tr, : len(SETUP_FEATURES)], y[tr])
        setup_test = fitted.predict_proba(x[te, : len(SETUP_FEATURES)])[:, 1]
    candidate_oof = records.filter(pa.array(~test))
    full_oof = candidate_oof["candidate_release_hist"].to_numpy()
    ablation = {
        "status": "diagnostic; not used to retune the frozen model on test",
        "model": "same HGB 200 iterations, 31 leaves, min leaf 100, L2 10",
        "setup_oof": metrics(y[tr], setup_oof),
        "release_oof": metrics(y[tr], full_oof),
        "setup_test": metrics(y[te], setup_test),
        "release_test": metrics(y[te], p[te]),
        "paired_match_ci": cluster_intervals(
            y[te],
            p[te],
            groups[te],
            meta["decision_threshold"],
            setup_test,
            repetitions=300,
        ),
    }
    matched = np.empty(len(table))
    matched[tr] = setup_oof
    matched[te] = setup_test
    field = "matched_setup_probability"
    idx = records.schema.get_field_index(field)
    records = (
        records.append_column(field, pa.array(matched))
        if idx < 0
        else records.set_column(idx, field, pa.array(matched))
    )
    pq.write_table(records, model_path.with_suffix(".validation.parquet"), compression="zstd")
    meta["predictions_sha256"] = sha256_file(model_path.with_suffix(".validation.parquet"))
    model_path.with_suffix(".metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    all_flashes = flash_table(
        root / "data" / "events.parquet",
        root / "data" / "release_context.parquet",
        known_only=False,
    )
    xx = matrix(all_flashes)
    known = ~np.isnan(all_flashes["impact"].to_numpy())
    g = all_flashes["match_id"].to_numpy()
    held = np.isin(g, meta["test_match_ids"])
    with threadpool_limits(limits=2):
        registration = HistGradientBoostingClassifier(
            max_iter=100,
            max_leaf_nodes=15,
            min_samples_leaf=150,
            l2_regularization=10,
            early_stopping=False,
            random_state=42,
        )
        registration.fit(xx[~held], known[~held])
        reg_p = registration.predict_proba(xx[held])[:, 1]
    missingness = {
        "all_flash_rows": len(all_flashes),
        "known_rows": int(known.sum()),
        "unknown_rows": int((~known).sum()),
        "known_fraction": float(known.mean()),
        "registered_effect_rate_bounds": [
            float(y.sum() / len(all_flashes)),
            float((y.sum() + (~known).sum()) / len(all_flashes)),
        ],
        "registration_prediction_test": metrics(known[held].astype(np.int8), reg_p),
        "interpretation": "Predictable registration suggests selection differences; no missing-at-random claim, no label imputation or inverse-probability causal correction",
    }
    errors = np.flatnonzero(test & ((p >= meta["decision_threshold"]) != y))
    ordered = errors[np.argsort(-np.abs(p[errors] - y[errors]))[:20]]
    examples = []
    for i in ordered:
        row = {
            key: table[key][int(i)].as_py()
            for key in [
                "source_row_index",
                "match_id",
                "player_id",
                "round_index",
                "side",
                "player_x",
                "player_y",
                "player_z",
                "pitch",
                "yaw",
                "impact",
                "enemy_blinded_players",
            ]
        }
        row["probability"] = float(p[i])
        examples.append(row)
    report = {
        "model_sha256": meta["artifact"]["sha256"],
        "data": meta["source"],
        "uncertainty": uncertainty,
        "selective_positive": selective,
        "slices": slices,
        "same_capacity_information_ablation": ablation,
        "missingness": missingness,
        "heldout_error_examples": examples,
        "external_evaluation": {
            "status": "not_available",
            "reason": "No independently obtained, matched Cache demos with player-eye/smoke telemetry have been verified",
        },
        "interpretation": "Internal, previously explored match holdout. Confidence intervals quantify match-sampling variation, not deployment validity or causal impact.",
    }
    save_analysis("model", report, root)
    print(
        json.dumps(
            {
                "test_ci": uncertainty["metrics"],
                "selective": selective,
                "ablation_setup_logloss": ablation["setup_oof"]["log_loss"],
                "registration_auc": missingness["registration_prediction_test"]["roc_auc"],
            },
            indent=2,
        ),
        flush=True,
    )
    return report


def triangle_segment_visible(origin, target, vertices, faces):
    """Independent vectorized Moller-Trumbore test against every mesh triangle."""
    origin, target = np.asarray(origin, float), np.asarray(target, float)
    vector = target - origin
    length = float(np.linalg.norm(vector))
    if not length:
        return True
    direction = vector / length
    triangle = vertices[faces].astype(np.float64)
    a = triangle[:, 0]
    e1 = triangle[:, 1] - a
    e2 = triangle[:, 2] - a
    h = np.cross(np.broadcast_to(direction, e2.shape), e2)
    det = np.sum(e1 * h, axis=1)
    valid = np.abs(det) > 1e-9
    inv = np.divide(1.0, det, out=np.zeros_like(det), where=valid)
    s = origin - a
    u = np.sum(s * h, axis=1) * inv
    q = np.cross(s, e1)
    v = np.sum(q * direction, axis=1) * inv
    t = np.sum(e2 * q, axis=1) * inv
    hit = (
        valid & (u >= -1e-9) & (v >= -1e-9) & (u + v <= 1 + 1e-9) & (t > 1e-4) & (t < length - 1e-4)
    )
    return not bool(hit.any())


def graph_at_resolution(geometry, areas, size):
    areas = areas.copy()
    areas["gx"] = np.floor(areas.centroid_x / size).astype(int)
    areas["gy"] = np.floor(areas.centroid_y / size).astype(int)
    areas["gz"] = np.floor(areas.centroid_z / 64).astype(int)
    areas = areas.sort_values("area_id").drop_duplicates(["gx", "gy", "gz"])
    points = areas[["centroid_x", "centroid_y", "centroid_z"]].to_numpy(float)
    points[:, 2] += 64
    pairs = cKDTree(points).query_pairs(1600, output_type="ndarray")
    pairs = pairs[np.linalg.norm(points[pairs[:, 0]] - points[pairs[:, 1]], axis=1) >= 128]
    clear = np.fromiter(
        (geometry.visible(points[a], points[b]) for a, b in pairs), bool, len(pairs)
    )
    pairs = pairs[clear]
    result = MapGeometry.__new__(MapGeometry)
    result.assets = geometry.assets
    result.manifest = geometry.manifest
    result.visibility = geometry.visibility
    result.a, result.b = points[pairs[:, 0]], points[pairs[:, 1]]
    result.midpoint = (result.a + result.b) / 2
    result.half_length = np.linalg.norm(result.b - result.a, axis=1) / 2
    result.tree = cKDTree(result.midpoint)
    result.search_radius = float(result.half_length.max()) + SMOKE_PARAMETERS["context_radius"]
    return result, {
        "xy_bin_units": size,
        "eye_points": len(points),
        "clear_segments": len(pairs),
    }


def diagnose_smoke(sample_size=2000):
    geometry = MapGeometry()
    areas = NavMesh(ASSETS / "de_cache.nav").areas.to_pandas()
    data = pq.read_table(
        ROOT / "data" / "events.parquet",
        filters=[("type", "=", "Smokes")],
        columns=["end_x", "end_y", "end_z"],
    )
    endpoints = np.column_stack([data["end_" + a].to_numpy() for a in "xyz"])
    nav = areas[["centroid_x", "centroid_y", "centroid_z"]].to_numpy(float)
    distances, _ = cKDTree(nav).query(endpoints)
    cells = np.unique(np.round(endpoints / 16) * 16, axis=0)
    rng = np.random.default_rng(817)
    sample = cells[rng.choice(len(cells), min(sample_size, len(cells)), replace=False)]
    variants = {128: geometry}
    graphs = [{"xy_bin_units": 128, "clear_segments": len(geometry.a), "eye_points": 715}]
    for size in [64, 256]:
        variants[size], info = graph_at_resolution(geometry, areas, size)
        graphs.append(info)
    values = {}
    for size, current in variants.items():
        rows = [current.smoke_at(point) for point in sample]
        values[str(size)] = np.array(
            [
                [
                    r["coverage_radius_96"],
                    r["potential_coverage"],
                    r["coverage_radius_192"],
                ]
                for r in rows
            ],
            dtype=float,
        )
        print("Smoke sampling sensitivity", size, flush=True)
    base = values["128"][:, 1]
    comparisons = []
    for size in [64, 256]:
        other = values[str(size)][:, 1]
        valid = np.isfinite(base) & np.isfinite(other)
        delta = np.abs(base[valid] - other[valid])
        comparisons.append(
            {
                "xy_bin_units": size,
                "paired_cells": int(valid.sum()),
                "spearman_rank_correlation": float(spearmanr(base[valid], other[valid]).statistic),
                "absolute_score_difference_quantiles": dict(
                    zip(
                        ["median", "p90", "p99"],
                        np.quantile(delta, [0.5, 0.9, 0.99]).tolist(),
                    )
                ),
            }
        )
    vertices, faces = read_mesh()
    pairs = [(geometry.a[i], geometry.b[i]) for i in rng.choice(len(geometry.a), 12, replace=False)]
    for _ in range(20):
        a, b = rng.choice(len(nav), 2, replace=False)
        pairs.append((nav[a] + [0, 0, 64], nav[b] + [0, 0, 64]))
    checks = []
    for a, b in pairs:
        actual = geometry.visible(a, b)
        reference = triangle_segment_visible(a, b, vertices, faces)
        checks.append(
            {
                "origin": a.tolist(),
                "target": b.tolist(),
                "bvh_visible": bool(actual),
                "triangle_visibility": reference,
                "agreement": bool(actual) == reference,
            }
        )
    valid = np.isfinite(values["128"]).all(axis=1)
    np.savez_compressed(
        ROOT / "reports" / "smoke_sensitivity.npz",
        endpoints=sample,
        coverage_64=values["64"],
        coverage_128=values["128"],
        coverage_256=values["256"],
    )
    report = {
        "map_client_version": geometry.manifest["client_version"],
        "sample": {
            "unit": "unique occupied 16-unit endpoint cell, equally weighted; not throw frequency",
            "seed": 817,
            "cells": len(sample),
            "population_cells": len(cells),
        },
        "coordinate_alignment": {
            "method": "Distance to nearest nav-area centroid; diagnostic only, not an exact walkable polygon or version match",
            "smoke_events": len(endpoints),
            "distance_quantiles_units": dict(
                zip(
                    ["median", "p90", "p99", "maximum"],
                    np.quantile(distances, [0.5, 0.9, 0.99, 1]).tolist(),
                )
            ),
            "fraction_within_128_units": float(np.mean(distances <= 128)),
            "fraction_within_256_units": float(np.mean(distances <= 256)),
            "source_game_version_compatibility": "unverified",
        },
        "graphs": graphs,
        "sampling_sensitivity": comparisons,
        "radius_sensitivity": {
            "radii_xy_units": [96, 144, 192],
            "monotonic_for_valid_cells": bool(
                np.all(np.diff(values["128"][valid], axis=1) >= -1e-12)
            ),
            "interpretation": "Sensitivity scenarios, not confidence intervals or calibrated smoke physics",
        },
        "independent_ray_validation": {
            "method": "Moller-Trumbore intersection against all 338375 triangles, compared with Rust BVH",
            "rays": len(checks),
            "all_agree": all(r["agreement"] for r in checks),
            "cases": checks,
        },
        "actual_smoke_validation": {
            "status": "not_available",
            "required": "Matched Cache demos, known map build and synchronized player-eye/smoke telemetry; renderer observations to calibrate occlusion",
        },
        "limitations": [
            "Static sampled potential is not enemy-specific or side-specific tactical value",
            "Changing graph resolution changes observer/target sampling weights",
            "Ellipsoid shape, density, growth, bullets and HE holes are uncalibrated",
            "No match winner, survival or causal impact is derived from this proxy",
        ],
    }
    save_analysis("smoke", report)
    print(
        json.dumps(
            {k: report[k] for k in ["coordinate_alignment", "sampling_sensitivity"]},
            indent=2,
        ),
        flush=True,
    )
    print(
        "Independent mesh-ray agreement",
        report["independent_ray_validation"]["all_agree"],
        flush=True,
    )
    return report


def audit(root=ROOT, quiet=False):
    root = Path(root)
    checks = {}
    source = json.loads((root / "data" / "events.metadata.json").read_text())
    events = pq.read_table(root / "data" / "events.parquet", columns=["match_id", "type", "impact"])
    context = pq.read_table(root / "data" / "release_context.parquet")
    cm = json.loads((root / "data" / "release_context.metadata.json").read_text())
    checks["complete_original_events"] = len(events) == source["rows"] == 2385417
    matches = set(pc.unique(events["match_id"]).to_pylist())
    checks["all_original_matches_accounted_for"] = (
        len(matches) == source["matches_with_events"]
        and len(matches) + len(source["empty_match_ids"]) == source["matches"] == 10000
        and not matches.intersection(source["empty_match_ids"])
    )
    checks["context_complete_and_ordered"] = len(context) == len(events) and np.array_equal(
        context["source_row_index"].to_numpy(), np.arange(len(events))
    )
    checks["context_provenance"] = (
        cm["events_sha256"] == sha256_file(root / "data" / "events.parquet")
        and cm["context_sha256"] == sha256_file(root / "data" / "release_context.parquet")
        and cm["source_sha256"] == source["source_sha256"]
    )
    for kind in ["Smokes", "Flashbangs", "HEs", "Fires"]:
        part = events.filter(pc.equal(events["type"], kind))
        checks[kind + "_source_counts"] = (
            len(part) == source["counts"][kind]
            and len(part) - part["impact"].null_count == source["counts"][kind + "/known"]
        )
    checks["redundant_proper_velocity_audited"] = (
        cm["proper_velocity_audit"]["identical_to_negative_player_velocity"] == len(events)
        and cm["proper_velocity_audit"]["maximum_residual_norm"] == 0
    )
    model_path = root / "models" / "flash.joblib"
    bundle, meta = load(model_path)
    table = flash_table(root / "data" / "events.parquet", root / "data" / "release_context.parquet")
    x = matrix(table)
    y = table["impact"].to_numpy().astype(np.int8)
    groups = table["match_id"].to_numpy()
    checks["model_inputs_match_provenance"] = (
        meta["source"]["events_sha256"] == sha256_file(root / "data" / "events.parquet")
        and meta["source"]["context_sha256"] == cm["context_sha256"]
    )
    checks["sample_fingerprint"] = fingerprint(x, y, groups) == meta["sample_sha256"]
    checks["feature_allowlist_excludes_outcomes"] = (
        meta["feature_allowlist"] == SETUP_FEATURES + RELEASE_FEATURES
        and meta["severity_features"] == FEATURES
        and meta["features"] in [FEATURES, SETUP_FEATURES]
    )
    trainids, testids = set(meta["train_match_ids"]), set(meta["test_match_ids"])
    checks["test_matches_are_disjoint"] = not trainids.intersection(
        testids
    ) and trainids | testids == set(groups)
    folds = [set(part) for part in meta["selection"]["fold_validation_match_ids"]]
    checks["five_training_only_folds"] = (
        len(folds) == 5
        and set.union(*folds) == trainids
        and sum(map(len, folds)) == len(trainids)
        and not any(f & testids for f in folds)
    )
    selected = min(
        meta["selection"]["candidates"],
        key=lambda key: (meta["selection"]["candidates"][key]["oof"]["log_loss"], key),
    )
    checks["selection_uses_training_oof_only"] = selected == meta["selected_model"]
    records = pq.read_table(model_path.with_suffix(".validation.parquet"))
    p = records["probability"].to_numpy()
    test = records["partition"].to_numpy() == "test"
    train = ~test
    checks["prediction_file_checksum"] = (
        sha256_file(model_path.with_suffix(".validation.parquet")) == meta["predictions_sha256"]
    )
    checks["prediction_identity_and_labels"] = len(records) == len(table) and all(
        records[key].equals(table[key])
        for key in ["source_row_index", "match_id", "projectile_id", "impact"]
    )
    checks["prediction_partition_matches_manifest"] = (
        set(groups[test]) == testids and set(groups[train]) == trainids
    )
    checks["finite_probabilities"] = bool(np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all())
    actual_p, actual_amount, actual_count = probabilities(bundle, x[test], partition="evaluation")
    checks["evaluation_bundle_reproduces_all_test_heads"] = all(
        np.allclose(actual, records[name].to_numpy()[test], rtol=0, atol=1e-12)
        for actual, name in [
            (actual_p, "probability"),
            (actual_amount, "conditional_flash_added"),
            (actual_count, "conditional_blinded_players"),
        ]
    )
    observed = metrics(y[test], p[test], meta["decision_threshold"])
    checks["reported_test_metrics_recompute"] = (
        all(
            np.isclose(observed[key], meta["test"]["metrics"][key], atol=1e-12)
            for key in [
                "roc_auc",
                "average_precision",
                "accuracy",
                "balanced_accuracy",
                "brier",
                "log_loss",
            ]
        )
        and observed["confusion_matrix"] == meta["test"]["metrics"]["confusion_matrix"]
    )
    candidates = records.filter(pa.array(train))
    checks["candidate_predictions_are_training_oof_only"] = all(
        np.isfinite(candidates["candidate_" + name].to_numpy()).all()
        and records.filter(pa.array(test))["candidate_" + name].null_count == int(test.sum())
        for name in meta["selection"]["candidates"]
    )
    checks["all_candidate_validation_losses_recompute"] = all(
        np.isclose(
            metrics(y[train], candidates["candidate_" + name].to_numpy())["log_loss"],
            value["oof"]["log_loss"],
            atol=1e-12,
        )
        for name, value in meta["selection"]["candidates"].items()
    )
    checks["training_predictions_are_selected_oof"] = np.allclose(
        p[train], candidates["candidate_" + selected].to_numpy(), rtol=0, atol=1e-12
    )
    checks["both_thresholds_use_training_only"] = (
        balanced_threshold(y[train], p[train]) == meta["decision_threshold"]
        and precision_threshold(y[train], p[train]) == meta["high_precision_threshold"]
    )
    counts = records["conditional_blinded_players"].to_numpy()
    amount = records["conditional_flash_added"].to_numpy()
    checks["count_head_respects_enemy_count_support"] = bool(((counts >= 1) & (counts <= 5)).all())
    checks["expectations_are_probability_times_conditional_mean"] = np.allclose(
        records["expected_blinded_players"].to_numpy(), p * counts, rtol=0, atol=1e-12
    ) and np.allclose(records["expected_flash_added"].to_numpy(), p * amount, rtol=0, atol=1e-12)
    deploy, _, _ = probabilities(bundle, x[test][:128], partition="deployment")
    checks["production_refit_separate_from_assessment"] = (
        meta["deployment"]["rows"] == len(table)
        and bool(np.any(np.abs(deploy - p[test][:128]) > 1e-8))
        and "setup_deployment" in bundle
    )
    verify_assets(root / "map_assets")
    geometry = MapGeometry(root / "map_assets")
    checks["static_map_geometry"] = (
        geometry.visibility.triangle_count == 338375 and len(geometry.a) == 18022
    )
    checks["all_potential_lines_unoccluded_without_smoke"] = all(
        geometry.visible(a, b) for a, b in zip(geometry.a, geometry.b)
    )
    scoremeta = json.loads((root / "data" / "grenade_scores.metadata.json").read_text())
    checks["current_scores_checksum"] = scoremeta["schema_version"] == 4 and scoremeta[
        "scores_sha256"
    ] == sha256_file(root / "data" / "grenade_scores.parquet")
    checks["current_scores_model_and_map_provenance"] = (
        scoremeta["model_sha256"] == meta["artifact"]["sha256"]
        and scoremeta["context_sha256"] == cm["context_sha256"]
        and scoremeta["map_manifest_sha256"] == sha256_file(root / "map_assets" / "manifest.json")
    )
    scores = pq.read_table(
        root / "data" / "grenade_scores.parquet",
        columns=[
            "type",
            "observed_impact",
            "outcome_status",
            "actual_enemy_view_blocked",
            "predicted_flash_probability",
            "expected_blinded_enemy_players",
            "expected_registered_flash_added",
            "flash_prediction_source",
            "potential_smoke_coverage",
            "smoke_coverage_radius96",
            "smoke_coverage_radius192",
            *["observed_" + name for name in BLIND_COUNTS],
            *[
                "observed_" + side + "_" + effect
                for side in ["enemy", "ally", "self"]
                for effect in ["damage", "flash_added"]
            ],
        ],
    )
    checks["all_source_events_scored"] = len(scores) == len(events)
    checks["unknown_labels_never_imputed"] = scores["observed_impact"].equals(events["impact"])
    effect_fields = [
        side + "_" + effect
        for side in ["enemy", "ally", "self"]
        for effect in ["damage", "flash_added"]
    ]
    effects = pq.read_table(root / "data/events.parquet", columns=effect_fields)
    effect_integrity = []
    for key in effect_fields:
        eligible_types = ["Flashbangs"] if "flash" in key else ["HEs", "Fires"]
        applicable = pc.and_(
            pc.is_valid(events["impact"]),
            pc.is_in(events["type"], value_set=pa.array(eligible_types)),
        )
        effect_integrity.append(
            scores["observed_" + key].equals(pc.if_else(applicable, effects[key], None))
        )
    expected_status = pc.if_else(
        pc.equal(events["type"], "Smokes"),
        "not_applicable_smoke",
        pc.if_else(
            pc.is_valid(events["impact"]),
            pc.if_else(pc.equal(events["impact"], 1), "known_positive", "known_zero"),
            "unknown",
        ),
    )
    checks["observed_effect_values_and_status_follow_grenade_type"] = all(
        effect_integrity
    ) and scores["outcome_status"].equals(expected_status)
    checks["enemy_view_not_fabricated"] = scores["actual_enemy_view_blocked"].null_count == len(
        scores
    )
    checks["blind_counts_preserved_from_raw_source"] = all(
        scores["observed_" + name].equals(context[name]) for name in BLIND_COUNTS
    )
    mask = pc.and_(pc.equal(scores["type"], "Flashbangs"), pc.is_valid(scores["observed_impact"]))
    known_scores = scores.filter(mask)
    checks["all_known_score_heads_use_crossfit_or_holdout"] = all(
        np.allclose(
            known_scores[name].to_numpy(),
            records[target].to_numpy(),
            rtol=0,
            atol=1e-12,
        )
        for name, target in [
            ("predicted_flash_probability", "probability"),
            ("expected_blinded_enemy_players", "expected_blinded_players"),
            ("expected_registered_flash_added", "expected_flash_added"),
        ]
    )
    origin = known_scores["flash_prediction_source"].to_numpy()
    checks["prediction_source_labels_are_correct"] = bool(
        np.all(origin[test] == "heldout_match") and np.all(origin[train] == "train_out_of_fold")
    )
    smokes = scores.filter(pc.equal(scores["type"], "Smokes"))
    lo = smokes["smoke_coverage_radius96"].to_numpy()
    base = smokes["potential_smoke_coverage"].to_numpy()
    hi = smokes["smoke_coverage_radius192"].to_numpy()
    valid = np.isfinite(base)
    checks["smoke_ratios_and_radius_sensitivity"] = bool(
        (
            (0 <= base[valid])
            & (base[valid] <= 1)
            & (lo[valid] <= base[valid] + 1e-12)
            & (base[valid] <= hi[valid] + 1e-12)
        ).all()
    )
    analysis = read_analysis(root)
    smoke, diagnostic = analysis["smoke"], analysis["model"]
    checks["independent_triangle_intersections_agree"] = (
        smoke["independent_ray_validation"]["all_agree"]
        and smoke["independent_ray_validation"]["rays"] == 32
    )
    checks["uncertainty_matches_evaluation_artifact"] = diagnostic["model_sha256"] == meta[
        "artifact"
    ]["sha256"] and np.isclose(
        diagnostic["uncertainty"]["metrics"]["roc_auc"]["estimate"],
        observed["roc_auc"],
        atol=1e-12,
    )
    profiles = pq.read_table(root / "data" / "player_utility_profiles.parquet")
    checks["player_profile_throw_totals"] = pc.sum(profiles["total_throws"]).as_py() == len(events)
    checks["player_profiles_do_not_invent_overall_rating"] = (
        "player_kpd" not in profiles.column_names and "overall_rating" not in profiles.column_names
    )
    report = {
        "ok": bool(all(checks.values())),
        "checks": {key: bool(value) for key, value in checks.items()},
        "counts": {
            "events": len(events),
            "known_flashes": len(table),
            "checks": len(checks),
        },
        "model": {
            "selected": selected,
            "test": meta["test"],
            "confidence": diagnostic["uncertainty"]["metrics"],
        },
        "limitations": [
            "Internal previously explored match holdout; no fresh external Cache validation",
            "Registered-outcome population; missing outcomes remain unknown",
            "Static smoke ellipsoid and map version compatibility remain uncalibrated",
            "No causal impact or universal player KPD is claimed",
        ],
    }
    path = root / "reports/checks.json"
    saved = json.loads(path.read_text()) if path.exists() else {}
    saved["audit"] = report
    path.write_text(json.dumps(saved, indent=2, ensure_ascii=False) + "\n")
    if not quiet:
        print(json.dumps(report, indent=2, allow_nan=False))
    return report
