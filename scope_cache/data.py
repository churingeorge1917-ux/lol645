"""Извлечение бросков, предыдущих бросков и зарегистрированного эффекта."""

from __future__ import annotations
import hashlib
import json
import math
import time
from collections import Counter, defaultdict
from itertools import groupby
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scope_cache import ROOT
from scope_cache.core import direction, enemy_effect, read_matches, vector, sha256_file

IDS = ["match_id", "player_id", "projectile_id", "type"]


INTS = [
    "round_index",
    "side",
    "is_crouching",
    "is_airborne",
    "start_tick",
    "end_tick",
    "impact",
]


FLOATS = [
    "player_x",
    "player_y",
    "player_z",
    "velocity_x",
    "velocity_y",
    "velocity_z",
    "view_dx",
    "view_dy",
    "view_dz",
    "speed",
    "pitch",
    "yaw",
    "end_x",
    "end_y",
    "end_z",
    "enemy_damage",
    "ally_damage",
    "self_damage",
    "enemy_flash_added",
    "ally_flash_added",
    "self_flash_added",
    "fire_area_xy",
]


EVENT_SCHEMA = pa.schema(
    [(name, pa.string()) for name in IDS]
    + [(name, pa.int64()) for name in INTS]
    + [(name, pa.float64()) for name in FLOATS]
)


def fire_area(grenade):
    hull = grenade.get("FinalInfernoInfo", {}).get("ConvexHull3D", {})
    vertices = hull.get("Vertices")
    if not vertices:
        return None
    from scipy.spatial import ConvexHull, QhullError

    try:
        points = [vector(v)[:2] for v in vertices]
        return float(ConvexHull(points).volume) if len(points) >= 3 else 0.0
    except (QhullError, TypeError, KeyError):
        return None


def event_record(match_id, grenade, kind):
    info = grenade["GeneralInfo"]
    point = info["ThrowPoint"]
    position, velocity = vector(point["Position"]), vector(point["Velocity"])
    pitch, yaw = float(point["ViewY"]), float(point["ViewX"])
    view = direction(pitch, yaw)
    effect, _ = enemy_effect(grenade, kind)
    row = {
        "match_id": str(match_id),
        "player_id": str(info["ThrowerID"]),
        "projectile_id": str(info["ProjectileID"]),
        "type": kind,
        "round_index": int(info["RoundIndexMatch"]),
        "side": int(info["Team"]),
        "is_crouching": int(point["IsCrouching"]),
        "is_airborne": int(point["IsAirborne"]),
        "start_tick": int(info["StartTick"]) if info.get("StartTick") is not None else None,
        "end_tick": int(info["EndTick"]) if info.get("EndTick") is not None else None,
        "pitch": pitch,
        "yaw": yaw,
        "speed": math.dist(velocity, (0, 0, 0)),
        **effect,
        "fire_area_xy": fire_area(grenade) if kind == "Fires" else None,
    }
    for prefix, values, axes in (
        ("player_", position, "xyz"),
        ("velocity_", velocity, "xyz"),
        ("view_d", view, "xyz"),
        ("end_", vector(info["PosTo"]), "xyz"),
    ):
        row.update({prefix + axis: value for axis, value in zip(axes, values)})
    if row["side"] not in (2, 3) or row["round_index"] < 0:
        raise ValueError("Invalid event side or round")
    if any(row[name] is not None and not math.isfinite(row[name]) for name in FLOATS):
        raise ValueError("Nonfinite event coordinate/effect")
    return row


def extract_events(source: Path, output: Path, expected_sha256=None):
    with source.open("rb") as stream:
        sha = hashlib.file_digest(stream, "sha256").hexdigest()
    if expected_sha256 is not None and sha != expected_sha256:
        raise ValueError("Original source checksum mismatch")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".building.parquet")
    count, buffer, matches, empty_matches = Counter(), [], set(), []
    writer = pq.ParquetWriter(
        temporary,
        EVENT_SCHEMA,
        compression="zstd",
        compression_level=9,
        use_dictionary=IDS + INTS,
        use_byte_stream_split=FLOATS,
    )
    start = time.monotonic()

    def collect(match):
        match_id = str(match["MatchID"])
        if match_id in matches:
            raise ValueError("Duplicate source match")
        matches.add(match_id)
        if not any(match[kind] for kind in ("Smokes", "Flashbangs", "HEs", "Fires")):
            empty_matches.append(match_id)
        seen = set()
        for kind in ("Smokes", "Flashbangs", "HEs", "Fires"):
            for grenade in match[kind]:
                info = grenade["GeneralInfo"]
                key = (info["RoundIndexMatch"], str(info["ProjectileID"]))
                if key in seen:
                    raise ValueError("Duplicate projectile within a round")
                seen.add(key)
                row = event_record(match_id, grenade, kind)
                buffer.append(row)
                count[kind] += 1
                count[kind + "/known"] += row["impact"] is not None
                count[kind + "/positive"] += row["impact"] == 1
                if len(buffer) >= 25000:
                    writer.write_table(pa.Table.from_pylist(buffer, schema=EVENT_SCHEMA))
                    buffer.clear()
        if len(matches) % 250 == 0:
            print(
                f"Extracted {len(matches)} matches / {sum(count[k] for k in ('Smokes', 'Flashbangs', 'HEs', 'Fires'))} events; {time.monotonic() - start:.0f}s",
                flush=True,
            )

    try:
        read_matches(source, collect)
        if buffer:
            writer.write_table(pa.Table.from_pylist(buffer, schema=EVENT_SCHEMA))
        writer.close()
        temporary.replace(output)
    except BaseException:
        writer.close()
        temporary.unlink(missing_ok=True)
        raise
    report = {
        "source_file": source.name,
        "source_bytes": source.stat().st_size,
        "source_sha256": sha,
        "matches": len(matches),
        "counts": dict(count),
        "matches_with_events": len(matches) - len(empty_matches),
        "empty_match_ids": sorted(empty_matches),
        "rows": sum(count[k] for k in ("Smokes", "Flashbangs", "HEs", "Fires")),
        "schema_version": 1,
        "event_file": output.name,
        "limitations": [
            "No synchronous all-player snapshots",
            "No round winners or independent game KPI",
            "Tick fields are original projectile ticks; smoke lifetime is not inferred",
            "Effect measurement units are preserved from the source",
        ],
    }
    output.with_suffix(".metadata.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    return report


KINDS = ("Smokes", "Flashbangs", "HEs", "Fires")


PAST_FEATURES = [f"past_{kind.lower()}_{side}" for kind in KINDS for side in ("own", "opponent")]


PAST_FEATURES += [
    "past_player_throws",
    "ticks_since_first_grenade",
    "ticks_since_own_grenade",
    "ticks_since_opponent_grenade",
]


RELEASE_FEATURES = [
    "release_offset_x",
    "release_offset_y",
    "release_offset_z",
] + PAST_FEATURES


BLIND_COUNTS = ["enemy_blinded_players", "ally_blinded_players", "self_blinded_players"]


CONTEXT_SCHEMA = pa.schema(
    [
        ("source_row_index", pa.int64()),
        ("type", pa.string()),
        *[(name, pa.float64()) for name in RELEASE_FEATURES],
        *[(name, pa.int64()) for name in BLIND_COUNTS],
    ]
)


def previous_activity(grenades):
    """Only releases at strictly smaller ticks in the same round are included."""
    rounds = defaultdict(list)
    for kind, grenade in grenades:
        info = grenade["GeneralInfo"]
        if info.get("StartTick") is not None:
            rounds[int(info["RoundIndexMatch"])].append((int(info["StartTick"]), kind, info))
    result = {}
    for observations in rounds.values():
        observations.sort(key=lambda x: x[0])
        first = observations[0][0]
        counts, own_player, latest = Counter(), Counter(), {}
        for tick, ties in groupby(observations, key=lambda x: x[0]):
            ties = list(ties)
            for _, _, info in ties:
                side = int(info["Team"])
                opponent = 5 - side
                values = {f"past_{kind.lower()}_own": counts[(kind, side)] for kind in KINDS}
                values.update(
                    {f"past_{kind.lower()}_opponent": counts[(kind, opponent)] for kind in KINDS}
                )
                values.update(
                    past_player_throws=own_player[str(info["ThrowerID"])],
                    ticks_since_first_grenade=tick - first,
                    ticks_since_own_grenade=tick - latest[side] if side in latest else -1,
                    ticks_since_opponent_grenade=tick - latest[opponent]
                    if opponent in latest
                    else -1,
                )
                result[(int(info["RoundIndexMatch"]), str(info["ProjectileID"]))] = values
            for _, kind, info in ties:
                side = int(info["Team"])
                counts[(kind, side)] += 1
                own_player[str(info["ThrowerID"])] += 1
                latest[side] = tick
    return result


def blind_counts(grenade, kind):
    if kind != "Flashbangs" or not isinstance(grenade.get("FlashedPlayers"), list):
        return {name: None for name in BLIND_COUNTS}
    groups = {"enemy": set(), "ally": set(), "self": set()}
    info = grenade["GeneralInfo"]
    for target in grenade["FlashedPlayers"]:
        added = target.get("FlashDurationAdded")
        if (
            target.get("Team") not in (2, 3)
            or not isinstance(added, (float, int))
            or not math.isfinite(added)
            or added < 0
        ):
            return {name: None for name in BLIND_COUNTS}
        group = (
            "self"
            if str(target["PlayerID"]) == str(info["ThrowerID"])
            else "ally"
            if target["Team"] == info["Team"]
            else "enemy"
        )
        if added > 0:
            groups[group].add(str(target["PlayerID"]))
    return {group + "_blinded_players": len(players) for group, players in groups.items()}


def extract_context(
    source,
    events=ROOT / "data" / "events.parquet",
    output=ROOT / "data" / "release_context.parquet",
):
    source, events, output = map(Path, (source, events, output))
    expected = json.loads(events.with_suffix(".metadata.json").read_text())
    if sha256_file(source) != expected["source_sha256"]:
        raise ValueError("Original source checksum differs from prepared events")
    expected_batches = pq.ParquetFile(events).iter_batches(
        batch_size=25000, columns=["match_id", "round_index", "projectile_id"]
    )
    temporary = output.with_suffix(".building.parquet")
    counts = Counter()
    buffer = []
    offset = 0
    start = time.monotonic()
    matches = set()
    worst_duplicate_error = 0.0
    writer = pq.ParquetWriter(
        temporary,
        CONTEXT_SCHEMA,
        compression="zstd",
        compression_level=9,
        use_dictionary=["type"],
        use_byte_stream_split=[f.name for f in CONTEXT_SCHEMA if pa.types.is_floating(f.type)],
    )

    def flush():
        if not buffer:
            return
        table = pa.Table.from_pylist(buffer, schema=CONTEXT_SCHEMA)
        wanted = pa.Table.from_batches([next(expected_batches)])
        if not all(
            pa.array([row[name] for row in buffer], type=wanted.schema.field(name).type).equals(
                wanted[name].combine_chunks()
            )
            for name in ["match_id", "round_index", "projectile_id"]
        ):
            raise ValueError("Raw-source ordering or identity differs from events.parquet")
        writer.write_table(table)
        buffer.clear()

    def collect(match):
        nonlocal offset, worst_duplicate_error
        mid = str(match["MatchID"])
        if mid in matches:
            raise ValueError("Duplicate match")
        matches.add(mid)
        grenades = [(kind, g) for kind in KINDS for g in match[kind]]
        past = previous_activity(grenades)
        for kind, g in grenades:
            info = g["GeneralInfo"]
            body = np.asarray(vector(info["ThrowPoint"]["Velocity"]))
            proper = np.asarray(vector(info["StartProperVelocity"]))
            error = float(np.linalg.norm(body + proper))
            worst_duplicate_error = max(worst_duplicate_error, error)
            counts["proper_equals_negative_player_velocity"] += error <= 1e-5
            release = np.asarray(vector(info["PosFrom"]))
            position = np.asarray(vector(info["ThrowPoint"]["Position"]))
            row = {
                "source_row_index": offset,
                "type": kind,
                "match_id": mid,
                "round_index": int(info["RoundIndexMatch"]),
                "projectile_id": str(info["ProjectileID"]),
                **dict(zip(["release_" + a for a in "xyz"], release)),
                **dict(zip(["recorded_proper_v" + a for a in "xyz"], proper)),
                **dict(zip(["release_offset_" + a for a in "xyz"], release - position)),
                **past.get(
                    (int(info["RoundIndexMatch"]), str(info["ProjectileID"])),
                    {key: None for key in PAST_FEATURES},
                ),
                **blind_counts(g, kind),
            }
            buffer.append(row)
            offset += 1
            counts[kind] += 1
            if len(buffer) == 25000:
                flush()
        if len(matches) % 500 == 0:
            print(
                f"Context: {len(matches)} matches / {offset} events / {time.monotonic() - start:.0f}s",
                flush=True,
            )

    try:
        read_matches(source, collect)
        flush()
        if next(expected_batches, None) is not None or offset != expected["rows"]:
            raise ValueError("Incomplete source coverage")
        writer.close()
        temporary.replace(output)
    except BaseException:
        writer.close()
        temporary.unlink(missing_ok=True)
        raise
    report = {
        "schema_version": 2,
        "rows": offset,
        "matches": len(matches),
        "source_sha256": expected["source_sha256"],
        "events_sha256": sha256_file(events),
        "context_sha256": sha256_file(output),
        "counts": dict(counts),
        "proper_velocity_audit": {
            "identical_to_negative_player_velocity": counts[
                "proper_equals_negative_player_velocity"
            ],
            "maximum_residual_norm": worst_duplicate_error,
            "interpretation": "Redundant field; must not be described as initial projectile velocity",
        },
        "feature_allowlist": RELEASE_FEATURES,
        "feature_time": "release; strictly previous throws in the same round, excluding equal-tick throws",
        "past_context_units": "original demo ticks; -1 means no previous throw on that side; no conversion to seconds",
        "limitations": [
            "Source does not record confirmed projectile launch speed or throw strength",
            "Release context is server-observed information, not necessarily known to the player",
            "Round activity clock starts at the first recorded grenade, not at round start",
            "Blind counts are retrospective outcomes, never model inputs",
        ],
    }
    output.with_suffix(".metadata.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps(report, indent=2), flush=True)
    return report
