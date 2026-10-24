"""Импорт демо, выживание и перекрытие обзора смоками."""

from __future__ import annotations
import csv
import gzip
import json
import math
import tempfile
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scope_cache.core import sha256_file, direction
from scope_cache.geometry import MapGeometry, SMOKE_PARAMETERS, smoke_chords

UPSTREAM = "https://github.com/Gary2005/cs-net"


UPSTREAM_COMMIT = "268dd2193c8a3d6e330c07dde5a851196c94eeaa"


SIDES = {"T": 2, "CT": 3}


UTILITY = {
    "hegrenade",
    "inferno",
    "molotov",
    "incgrenade",
    "flashbang",
    "smokegrenade",
    "decoy",
}


ROUND_SCHEMA = pa.schema(
    [
        ("match_id", pa.string()),
        ("player_id", pa.string()),
        ("round_index", pa.int64()),
        ("demo_round_id", pa.int64()),
        ("side", pa.int64()),
        ("round_won", pa.int8()),
        ("alive_end", pa.int8()),
        ("enemy_kills", pa.int64()),
        ("deaths", pa.int64()),
        ("enemy_assists", pa.int64()),
        ("enemy_damage", pa.float64()),
        ("weapon_enemy_damage", pa.float64()),
        ("distance_travelled", pa.float64()),
        ("movement_observed_seconds", pa.float64()),
        ("maximum_sample_gap_seconds", pa.float64()),
    ]
)


STATE_SCHEMA = pa.schema(
    [
        ("match_id", pa.string()),
        ("player_id", pa.string()),
        ("round_index", pa.int64()),
        ("side", pa.int64()),
        ("tick", pa.int64()),
        ("round_seconds", pa.float64()),
        *[
            (name, pa.float64())
            for name in [
                "x",
                "y",
                "z",
                "yaw",
                "pitch",
                "eye_x",
                "eye_y",
                "eye_z",
                "gaze_x",
                "gaze_y",
                "gaze_z",
            ]
        ],
    ]
)


SMOKE_SCHEMA = pa.schema(
    [
        ("match_id", pa.string()),
        ("round_index", pa.int64()),
        ("smoke_index", pa.int64()),
        *[(name, pa.float64()) for name in ["start_seconds", "end_seconds", "x", "y", "z"]],
        ("thrower_id", pa.string()),
        ("projectile_id", pa.string()),
    ]
)


def exact_id(value):
    if isinstance(value, (float, np.floating, bool)) or not isinstance(
        value, (str, int, np.integer)
    ):
        raise ValueError("Expected an exact string/integer identifier; float IDs are lossy")
    text = str(value)
    if not text:
        raise ValueError("Empty identifier")
    return text


def load_demo(path):
    opener = gzip.open if Path(path).suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        result = json.load(stream)
    if result.get("format") != "cs2.demo.v2":
        raise ValueError("Expected CS-NET cs2.demo.v2 JSON or JSON.gz")
    for player in result["players"]:
        player["steamid"] = exact_id(player["steamid"])
    for rd in result["rounds"]:
        for smoke in rd.get("events", {}).get("smokes", []):
            for key in ["thrower_id", "projectile_id"]:
                if smoke.get(key) is not None:
                    smoke[key] = exact_id(smoke[key])
    return result


def finite(values):
    return all(value is not None and math.isfinite(float(value)) for value in values)


def eye_height_at(player, index, default):
    values = player.get("eye_height")
    height = values[index] if values is not None and values[index] is not None else default
    if not finite([height]) or not 0 < float(height) < 128:
        raise ValueError("Invalid player eye height")
    return float(height)


def account32(steamid):
    value = int(exact_id(steamid)) - 76561197960265728
    if not 0 < value < 2**32:
        raise ValueError("Expected an individual SteamID64 for account32 conversion")
    return str(value)


def validate_round(round_data, player_count):
    ticks, seconds = round_data["ticks"], round_data["round_seconds"]
    if not ticks or len(ticks) != len(seconds) or not finite(seconds):
        raise ValueError("Round requires parallel sampled ticks and round_seconds")
    if any(b <= a for a, b in zip(ticks, ticks[1:])) or any(
        b <= a for a, b in zip(seconds, seconds[1:])
    ):
        raise ValueError("Round ticks and seconds must increase strictly")
    if len(round_data["players"]) != player_count or len(round_data["teams"]) != player_count:
        raise ValueError("Round roster does not match the demo header")
    for player in round_data["players"]:
        for key in ["x", "y", "z", "yaw", "pitch", "alive"]:
            if len(player[key]) != len(ticks):
                raise ValueError(f"Unaligned player {key} array; exact synchronization is required")
        if "eye_height" in player and len(player["eye_height"]) != len(ticks):
            raise ValueError("Unaligned player eye-height array")


def round_records(demo, round_data, match_id, round_offset=0, player_map=None, max_gap=0.5):
    roster, players = demo["players"], round_data["players"]
    validate_round(round_data, len(roster))
    teams, times = round_data["teams"], np.asarray(round_data["round_seconds"], float)
    events = round_data.get("events", {})
    kills, damage = events.get("kills"), events.get("damage")
    if round_data.get("end_tick") is not None:
        boundary = int(round_data["end_tick"])
        begin = round_data.get("begin_tick")

        def in_round(values):
            return (
                [
                    v
                    for v in values
                    if int(v["t"]) <= boundary and (begin is None or int(v["t"]) >= int(begin))
                ]
                if values is not None
                else None
            )

        kills, damage = in_round(kills), in_round(damage)
    completed = round_data.get("winner") in SIDES and bool(round_data.get("end_reason"))
    weapon_names = {
        int(index): name.removeprefix("weapon_") for name, index in demo.get("weapons", {}).items()
    }
    rid = int(round_data["id"]) + round_offset
    if rid < 0:
        raise ValueError("Round offset produces a negative round index")
    records = []
    for index, (identity, player, team) in enumerate(zip(roster, players, teams)):
        if team not in SIDES:
            if any(player["alive"]):
                raise ValueError("Alive player has unknown side")
            continue
        steamid = str(identity["steamid"])
        pid = player_map[steamid] if player_map is not None else steamid
        states = player["alive"]
        died = kills is not None and any(k.get("v") == index for k in kills)
        alive_end = (
            0
            if died or (states and states[-1] is False)
            else (1 if completed and kills is not None and any(states) else None)
        )
        disconnected = any(event.get("p") == index for event in events.get("disconnects", []))
        if not died and (disconnected or (states and states[-1] is None)):
            alive_end = None
        count = Counter()
        if kills is not None:
            for kill in kills:
                a, v, assister = (
                    kill.get("a", -1),
                    kill.get("v", -1),
                    kill.get("as", -1),
                )
                count["deaths"] += v == index
                enemy = isinstance(v, int) and 0 <= v < len(teams) and teams[v] != team
                count["enemy_kills"] += a == index and enemy
                count["enemy_assists"] += assister == index and a != index and enemy
        enemy_damage = weapon_damage = None
        if damage is not None:
            enemy_damage, weapon_damage = 0.0, 0.0
            weapon_known = True
            for hit in damage:
                v = hit.get("v", -1)
                if (
                    hit.get("a") != index
                    or not isinstance(v, int)
                    or not 0 <= v < len(teams)
                    or teams[v] == team
                ):
                    continue
                amount = float(hit["hp"])
                if not math.isfinite(amount) or amount < 0:
                    raise ValueError("Invalid recorded damage")
                enemy_damage += amount
                name = weapon_names.get(hit.get("w"))
                if name is None:
                    weapon_known = False
                elif name not in UTILITY and name not in {
                    "world",
                    "worldspawn",
                    "planted_c4",
                }:
                    weapon_damage += amount
            if not weapon_known:
                weapon_damage = None
        distance = observed = 0.0
        for t in range(len(times) - 1):
            dt = times[t + 1] - times[t]
            points = [[player[axis][j] for axis in "xyz"] for j in (t, t + 1)]
            if states[t] and states[t + 1] and dt <= max_gap and finite(points[0] + points[1]):
                distance += math.dist(*points)
                observed += dt
        records.append(
            {
                "match_id": str(match_id),
                "player_id": str(pid),
                "round_index": rid,
                "demo_round_id": int(round_data["id"]),
                "side": SIDES[team],
                "round_won": int(team == round_data["winner"])
                if round_data.get("winner") in SIDES
                else None,
                "alive_end": alive_end,
                **{
                    key: count[key] if kills is not None else None
                    for key in ["enemy_kills", "deaths", "enemy_assists"]
                },
                "enemy_damage": enemy_damage,
                "weapon_enemy_damage": weapon_damage,
                "distance_travelled": distance,
                "movement_observed_seconds": observed,
                "maximum_sample_gap_seconds": float(np.diff(times).max())
                if len(times) > 1
                else 0.0,
            }
        )
    return records


def sampled_states(demo, round_data, match_id, round_offset, player_map, eye_height):
    for index, player in enumerate(round_data["players"]):
        steamid = str(demo["players"][index]["steamid"])
        pid = player_map[steamid] if player_map is not None else steamid
        for t, tick in enumerate(round_data["ticks"]):
            values = [player[key][t] for key in ["x", "y", "z", "yaw", "pitch"]]
            if not player["alive"][t] or not finite(values):
                continue
            x, y, z, yaw, pitch = map(float, values)
            gaze = direction(pitch, yaw)
            yield {
                "match_id": str(match_id),
                "player_id": str(pid),
                "round_index": int(round_data["id"]) + round_offset,
                "side": SIDES[round_data["teams"][index]],
                "tick": int(tick),
                "round_seconds": float(round_data["round_seconds"][t]),
                "x": x,
                "y": y,
                "z": z,
                "yaw": yaw,
                "pitch": pitch,
                "eye_x": x,
                "eye_y": y,
                "eye_z": z + eye_height_at(player, t, eye_height),
                **dict(zip(["gaze_x", "gaze_y", "gaze_z"], gaze)),
            }


def smoke_exposure(round_data, geometry, *, eye_height=64.0, fov_degrees=90.0, max_gap=0.5):
    if not 0 < fov_degrees < 180 or max_gap <= 0 or not math.isfinite(eye_height):
        raise ValueError("Invalid FOV, eye height or maximum sample gap")
    smokes = round_data.get("events", {}).get("smokes", [])
    result = [
        {
            "smoke_index": i,
            "thrower_id": smoke.get("thrower_id"),
            "projectile_id": smoke.get("projectile_id"),
            **{
                f"{metric}_{side}": 0.0
                for metric in [
                    "eligible_pair_seconds",
                    "blocked_pair_seconds",
                    "marginal_blocked_pair_seconds",
                ]
                for side in ["T", "CT"]
            },
            "sampled_active_seconds": 0.0,
        }
        for i, smoke in enumerate(smokes)
    ]
    times = round_data["round_seconds"]
    for t in range(len(times) - 1):
        begin, end = float(times[t]), float(times[t + 1])
        if end - begin > max_gap:
            continue
        boundaries = sorted(
            {
                begin,
                end,
                *[
                    float(s[key])
                    for s in smokes
                    for key in ["ts", "te"]
                    if s.get(key) is not None and begin < float(s[key]) < end
                ],
            }
        )
        pairs = []
        for i, observer in enumerate(round_data["players"]):
            if not observer["alive"][t]:
                continue
            eye = np.array([observer[a][t] for a in "xyz"], float) + [
                0,
                0,
                eye_height_at(observer, t, eye_height),
            ]
            angles = [observer["pitch"][t], observer["yaw"][t]]
            if not finite([*eye, *angles]):
                continue
            gaze = np.asarray(direction(*angles))
            for j, target in enumerate(round_data["players"]):
                if round_data["teams"][i] == round_data["teams"][j] or not target["alive"][t]:
                    continue
                point = np.array([target[a][t] for a in "xyz"], float) + [
                    0,
                    0,
                    eye_height_at(target, t, eye_height),
                ]
                delta = point - eye
                if not finite(point) or np.linalg.norm(delta) == 0:
                    continue
                if np.dot(gaze, delta) / np.linalg.norm(delta) >= math.cos(
                    math.radians(fov_degrees / 2)
                ) and geometry.visible(eye, point):
                    pairs.append((eye, point, round_data["teams"][i]))
        for a, b in zip(boundaries, boundaries[1:]):
            midpoint = (a + b) / 2
            active = [
                i
                for i, s in enumerate(smokes)
                if float(s["ts"]) <= midpoint and (s.get("te") is None or midpoint < float(s["te"]))
            ]
            if not active:
                continue
            masks = []
            for i in active:
                center = np.array([smokes[i][axis] for axis in "xyz"], float) + [
                    0,
                    0,
                    SMOKE_PARAMETERS["center_height"],
                ]
                masks.append(
                    np.asarray(
                        [
                            float(smoke_chords(eye, point, center))
                            >= SMOKE_PARAMETERS["minimum_chord"]
                            for eye, point, _ in pairs
                        ],
                        bool,
                    )
                )
            total = np.sum(masks, axis=0) if pairs else np.array([], int)
            for i, mask in zip(active, masks):
                result[i]["sampled_active_seconds"] += b - a
                for side in ["T", "CT"]:
                    own = np.array([p[2] == side for p in pairs], bool)
                    result[i][f"eligible_pair_seconds_{side}"] += int(own.sum()) * (b - a)
                    result[i][f"blocked_pair_seconds_{side}"] += int((own & mask).sum()) * (b - a)
                    result[i][f"marginal_blocked_pair_seconds_{side}"] += int(
                        (own & mask & (total == 1)).sum()
                    ) * (b - a)
    return result


def join_events(events, context):
    keys = ["match_id", "round_index", "player_id", "side"]
    identities = list(zip(*(context[key].to_pylist() for key in keys)))
    if len(set(identities)) != len(identities):
        raise ValueError("Duplicate player-round context")
    indexed = events.append_column("_event_row", pa.array(np.arange(len(events))))
    joined = indexed.join(context, keys=keys, join_type="left outer").sort_by(
        [("_event_row", "ascending")]
    )
    return joined.drop(["_event_row"])


def _import_demo(
    path,
    output,
    *,
    match_id,
    expected_map="de_cache",
    round_offset=0,
    player_map=None,
    eye_height=64.0,
    max_gap=0.5,
    fov_degrees=90.0,
    score_smokes=False,
    kpd=None,
    event_path=None,
    id_scheme="steam64",
    allow_geometry_mismatch=False,
):
    demo = load_demo(path)
    if demo.get("map") != expected_map:
        raise ValueError(
            f"Map mismatch: got {demo.get('map')}, expected {expected_map}; maps must not be mixed"
        )
    ids = [str(p["steamid"]) for p in demo["players"]]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate demo player identifiers")
    if id_scheme not in ["steam64", "steam-account32"]:
        raise ValueError("Unknown player ID scheme")
    if id_scheme == "steam-account32":
        if player_map is not None:
            raise ValueError("Choose explicit mapping or account32 conversion, not both")
        player_map = {pid: account32(pid) for pid in ids}
    if player_map is not None:
        player_map = {exact_id(k): exact_id(v) for k, v in player_map.items()}
    for rd in demo["rounds"]:
        for smoke in rd.get("events", {}).get("smokes", []):
            owner = smoke.get("thrower_id")
            if owner is not None:
                owner = str(owner)
                if owner not in ids:
                    raise ValueError("Smoke owner is absent from the demo roster")
                smoke["thrower_id"] = str(player_map[owner]) if player_map else owner
    if player_map is not None and (
        set(player_map) != set(ids) or len(set(player_map.values())) != len(ids)
    ):
        raise ValueError("Player mapping must cover the full roster one-to-one")
    if not match_id or max_gap <= 0:
        raise ValueError("An explicit match identifier and positive max_gap are required")
    if score_smokes and expected_map != "de_cache":
        raise ValueError("Included collision geometry supports only de_cache")
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Choose an empty output directory to avoid mixing demos")
    output.mkdir(parents=True, exist_ok=True)
    geometry = MapGeometry() if score_smokes else None
    observed_patch = str(demo.get("source", {}).get("header", {}).get("patch_version", ""))
    expected_patch = geometry.manifest["patch_version"].replace(".", "") if geometry else None
    compatibility = (
        "not_used"
        if not geometry
        else "unverified_no_demo_patch"
        if not observed_patch
        else "patch_matches"
        if observed_patch == expected_patch
        else "patch_mismatch"
    )
    if compatibility == "patch_mismatch" and not allow_geometry_mismatch:
        raise ValueError(
            f"Demo patch {observed_patch} differs from geometry patch {expected_patch}; supply matching map assets or explicitly allow experimental mismatch"
        )
    records, smokes, exposure, round_ids = [], [], [], set()
    with pq.ParquetWriter(
        output / "player_states.parquet", STATE_SCHEMA, compression="zstd"
    ) as writer:
        for round_data in demo["rounds"]:
            if round_data["id"] in round_ids:
                raise ValueError("Duplicate round identifier")
            round_ids.add(round_data["id"])
            records.extend(
                round_records(demo, round_data, match_id, round_offset, player_map, max_gap)
            )
            states = list(
                sampled_states(demo, round_data, match_id, round_offset, player_map, eye_height)
            )
            writer.write_table(pa.Table.from_pylist(states, schema=STATE_SCHEMA))
            for i, smoke in enumerate(round_data.get("events", {}).get("smokes", [])):
                if not finite([smoke[k] for k in ["ts", "x", "y", "z"]]) or (
                    smoke.get("te") is not None
                    and (not finite([smoke["te"]]) or smoke["te"] < smoke["ts"])
                ):
                    raise ValueError("Invalid smoke interval")
                smokes.append(
                    {
                        "match_id": str(match_id),
                        "round_index": int(round_data["id"]) + round_offset,
                        "smoke_index": i,
                        "start_seconds": smoke["ts"],
                        "end_seconds": smoke.get("te"),
                        **{axis: smoke[axis] for axis in "xyz"},
                        "thrower_id": smoke.get("thrower_id"),
                        "projectile_id": smoke.get("projectile_id"),
                    }
                )
            if score_smokes:
                for row in smoke_exposure(
                    round_data,
                    geometry,
                    eye_height=eye_height,
                    fov_degrees=fov_degrees,
                    max_gap=max_gap,
                ):
                    exposure.append(
                        {
                            "match_id": str(match_id),
                            "round_index": int(round_data["id"]) + round_offset,
                            **row,
                        }
                    )
    context = pa.Table.from_pylist(records, schema=ROUND_SCHEMA)
    pq.write_table(context, output / "round_outcomes.parquet", compression="zstd")
    pq.write_table(
        pa.Table.from_pylist(smokes, schema=SMOKE_SCHEMA),
        output / "smoke_intervals.parquet",
        compression="zstd",
    )
    if score_smokes:
        (output / "smoke_exposure.json").write_text(
            json.dumps(exposure, indent=2, allow_nan=False) + "\n"
        )
    if kpd is not None:
        if kpd != "weapon-enemy-damage":
            raise ValueError("Only the explicit weapon-enemy-damage KPI is supported")
        with (output / "round_context.csv").open("w", newline="") as stream:
            fields = [
                "match_id",
                "player_id",
                "round_index",
                "side",
                "round_won",
                "player_kpd",
            ]
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for row in records:
                if row["round_won"] is not None and row["weapon_enemy_damage"] is not None:
                    writer.writerow(
                        {
                            **{key: row[key] for key in fields if key != "player_kpd"},
                            "player_kpd": row["weapon_enemy_damage"],
                        }
                    )
    report = {
        "upstream": UPSTREAM,
        "reviewed_commit": UPSTREAM_COMMIT,
        "input_sha256": sha256_file(path),
        "map": expected_map,
        "match_id": str(match_id),
        "round_offset": round_offset,
        "id_scheme": id_scheme,
        "geometry_compatibility": compatibility,
        "eye_height_sources": sorted(
            {
                p.get("eye_height_source", "fixed configured assumption")
                for rd in demo["rounds"]
                for p in rd["players"]
            }
        ),
        "smokes_with_recorded_owner": sum(s["thrower_id"] is not None for s in smokes),
        "rounds": len(round_ids),
        "player_rounds": len(records),
        "alive_end_known": sum(r["alive_end"] is not None for r in records),
        "smoke_intervals": len(smokes),
        "kpd_definition": "Observed enemy HP damage from weapons excluding grenades; retrospective, not a universal player rating"
        if kpd
        else None,
        "smoke_scoring": bool(score_smokes),
        "assumptions": {
            "eye_height": eye_height,
            "cone_fov_degrees": fov_degrees,
            "max_sample_gap_seconds": max_gap,
        },
        "limitations": [
            "Eye height uses supplied per-tick values when present, otherwise the configured assumption; cone FOV is assumed",
            "Smoke volume is approximate; static geometry may differ from the demo's game version",
            "Original CS-NET intervals omit projectile/thrower IDs; explicit parser extensions are retained, never guessed",
            "Survival, winner and damage are retrospective outcomes and must not become prethrow features",
            "An observed association with survival or victory is not a causal grenade effect",
            "No CS-NET checkpoint is loaded; no model probability is used as an observed label",
        ],
    }
    if event_path is not None:
        if expected_map != "de_cache":
            raise ValueError("Cannot join a different map to the Cache grenade dataset")
        events = pq.read_table(event_path)
        joined = join_events(events, context)
        pq.write_table(joined, output / "grenade_round_context.parquet", compression="zstd")
        report["joined_event_rows"] = len(joined) - joined["demo_round_id"].null_count
        report["event_rows"] = len(joined)
    (output / "import_report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def import_demo(path, output, **kwargs):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Choose an empty output directory to avoid mixing demos")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=output.name + ".building-", dir=output.parent
    ) as temporary:
        staged = Path(temporary) / "result"
        report = _import_demo(path, staged, **kwargs)
        if output.exists():
            output.rmdir()
        staged.replace(output)
    return report


def steamid(value):
    if value is None or pd.isna(value):
        return None
    if isinstance(value, (float, np.floating)):
        raise ValueError("A SteamID64 was supplied as a lossy float; exact identity is required")
    text = str(value)
    return text if text.isdigit() and int(text) > 0 else None


def clean_number(value):
    return (
        float(value)
        if value is not None and not pd.isna(value) and math.isfinite(float(value))
        else None
    )


def convert(path, output, *, expected_map="de_cache", sample_ticks=16, max_rounds=None):
    from demoparser2 import DemoParser

    path, output = Path(path), Path(output)
    if sample_ticks <= 0:
        raise ValueError("Sample tick interval must be positive")
    if max_rounds is not None and max_rounds <= 0:
        raise ValueError("max_rounds must be positive")
    parser = DemoParser(str(path))
    header = parser.parse_header()
    if header.get("map_name") != expected_map:
        raise ValueError(f"Demo map is {header.get('map_name')}, expected {expected_map}")
    if not str(header.get("demo_file_stamp", "")).startswith("PBDEMS2"):
        raise ValueError("Only Source 2 demos are supported")
    available = set(parser.list_game_events())

    def event(name):
        if name not in available:
            return pd.DataFrame()
        return parser.parse_event(name, other=["total_rounds_played", "game_time"])

    starts = event("round_start")
    ends = event("round_end")
    freezes = event("round_freeze_end")
    if starts.empty or ends.empty:
        raise ValueError("Complete round_start and round_end events are required")
    intervals = []
    for _, end in ends.sort_values("tick").iterrows():
        rid = int(end["total_rounds_played"])
        start = starts[(starts.total_rounds_played == rid) & (starts.tick <= end["tick"])]
        if start.empty:
            continue
        start = start.sort_values("tick").iloc[-1]
        freeze = (
            freezes[
                (freezes.total_rounds_played == rid)
                & (freezes.tick >= start["tick"])
                & (freezes.tick <= end["tick"])
            ]
            if not freezes.empty
            else pd.DataFrame()
        )
        begin = freeze.sort_values("tick").iloc[0] if not freeze.empty else start
        if int(end["tick"]) <= int(begin["tick"]):
            continue
        intervals.append(
            (
                rid,
                int(begin["tick"]),
                int(end["tick"]),
                float(begin["game_time"]),
                int(end["winner"]),
                int(end["reason"]),
            )
        )
    if max_rounds is not None:
        intervals = intervals[:max_rounds]
    if not intervals:
        raise ValueError("No complete playable rounds found")
    requested = sorted(
        {tick for _, a, b, *_ in intervals for tick in [*range(a, b + 1, sample_ticks), b]}
    )
    required = ["X", "Y", "Z", "yaw", "pitch", "is_alive", "team_num", "game_time"]
    states = parser.parse_ticks(required + ["duck_amount"], ticks=requested)
    missing = set(required + ["tick", "steamid"]) - set(states.columns)
    if missing:
        raise ValueError("Parser omitted required telemetry: " + ", ".join(sorted(missing)))
    states = states.copy()
    states["pid"] = [steamid(value) for value in states.steamid]
    states = states[states.pid.notna()].copy()
    if states.empty:
        raise ValueError("No identified player states")
    ids = sorted(set(states.pid), key=int)
    index = {pid: i for i, pid in enumerate(ids)}
    roster = []
    for pid in ids:
        names = states.loc[states.pid == pid, "name"].dropna() if "name" in states else []
        roster.append({"steamid": pid, "name": str(names.iloc[-1]) if len(names) else pid})
    kills = event("player_death")
    damage = event("player_hurt")
    smoke_start = event("smokegrenade_detonate")
    smoke_end = event("smokegrenade_expired")
    disconnects = event("player_disconnect")
    weapon_names = sorted(
        set(
            str(w) for frame in [kills, damage] if not frame.empty for w in frame["weapon"].dropna()
        )
    )
    weapons = {name: i for i, name in enumerate(weapon_names)}

    def player_index(value):
        pid = steamid(value)
        return index.get(pid, -1)

    def inside(frame, a, b):
        return frame[(frame.tick >= a) & (frame.tick <= b)] if not frame.empty else frame

    rounds = []
    dropped_time_samples = 0
    for rid, a, b, base_time, winner, reason in intervals:
        current = states[(states.tick >= a) & (states.tick <= b)].copy()
        time_by_tick = current.groupby("tick").game_time.first().sort_index()
        time_by_tick = time_by_tick[~time_by_tick.duplicated(keep="last")]
        ticks = [int(v) for v in time_by_tick.index]
        times = [float(v) - base_time for v in time_by_tick]
        if len(ticks) < 2:
            continue
        if any(q <= p for p, q in zip(times, times[1:])):
            raise ValueError("Nonmonotone recorded game clock")
        dropped_time_samples += len(set(current.tick)) - len(ticks)
        players = []
        teams = []
        for pid in ids:
            observed = (
                current[current.pid == pid]
                .sort_values("tick")
                .drop_duplicates("tick", keep="last")
                .set_index("tick")
            )
            side = observed.team_num.dropna()
            side = side[side.isin([2, 3])]
            teams.append(
                "T" if len(side) and int(side.iloc[-1]) == 2 else "CT" if len(side) else "?"
            )
            aligned = observed.reindex(ticks)
            player = {
                key: [clean_number(v) for v in aligned[col]]
                for key, col in [
                    ("x", "X"),
                    ("y", "Y"),
                    ("z", "Z"),
                    ("yaw", "yaw"),
                    ("pitch", "pitch"),
                ]
            }
            player["alive"] = [bool(v) if not pd.isna(v) else None for v in aligned.is_alive]
            duck = aligned.duck_amount if "duck_amount" in aligned else [None] * len(ticks)
            player["eye_height"] = [
                64.0 - 18.0 * float(np.clip(v, 0, 1)) if clean_number(v) is not None else 64.0
                for v in duck
            ]
            player["eye_height_source"] = (
                "assumed_from_duck_amount_64_to_46_units; not a measured camera origin"
            )
            players.append(player)
        timeline = {"kills": [], "damage": [], "smokes": [], "disconnects": []}
        for _, row in inside(kills, a, b).iterrows():
            timeline["kills"].append(
                {
                    "t": int(row["tick"]),
                    "a": player_index(row.get("attacker_steamid")),
                    "v": player_index(row.get("user_steamid")),
                    "as": player_index(row.get("assister_steamid")),
                    "w": weapons[str(row["weapon"])],
                    "ts": bool(row.get("thrusmoke", False)),
                }
            )
        for _, row in inside(damage, a, b).iterrows():
            timeline["damage"].append(
                {
                    "t": int(row["tick"]),
                    "a": player_index(row.get("attacker_steamid")),
                    "v": player_index(row.get("user_steamid")),
                    "hp": float(row["dmg_health"]),
                    "w": weapons[str(row["weapon"])],
                }
            )
        for _, row in inside(disconnects, a, b).iterrows():
            timeline["disconnects"].append(
                {"t": int(row["tick"]), "p": player_index(row.get("user_steamid"))}
            )
        for _, row in inside(smoke_start, a, b).iterrows():
            eid = int(row["entityid"])
            end = (
                smoke_end[(smoke_end.entityid == eid) & (smoke_end.tick >= row["tick"])]
                if not smoke_end.empty
                else pd.DataFrame()
            )
            next_start = smoke_start[
                (smoke_start.entityid == eid) & (smoke_start.tick > row["tick"])
            ]
            if not next_start.empty and not end.empty:
                end = end[end.tick < next_start.tick.min()]
            expiry = end.sort_values("tick").iloc[0] if not end.empty else None
            owner = steamid(row.get("user_steamid"))
            if owner not in index:
                owner = None
            timeline["smokes"].append(
                {
                    "ts": float(row["game_time"]) - base_time,
                    "te": float(expiry["game_time"]) - base_time if expiry is not None else None,
                    **{axis: float(row[axis]) for axis in "xyz"},
                    "thrower_id": owner,
                    "projectile_id": f"entity:{eid}:detonation:{int(row['tick'])}",
                    "engine_entity_id": eid,
                    "detonation_tick": int(row["tick"]),
                }
            )
        rounds.append(
            {
                "id": rid,
                "teams": teams,
                "winner": {2: "T", 3: "CT"}.get(winner, ""),
                "end_reason": f"engine_round_end_reason_{reason}",
                "begin_tick": a,
                "end_tick": b,
                "ticks": ticks,
                "round_seconds": times,
                "players": players,
                "events": timeline,
            }
        )
    data = {
        "format": "cs2.demo.v2",
        "map": expected_map,
        "players": roster,
        "weapons": weapons,
        "places": {},
        "rounds": rounds,
        "source": {
            "demo_sha256": sha256_file(path),
            "header": header,
            "parser": "demoparser2",
            "parser_version": __import__("importlib.metadata", fromlist=["version"]).version(
                "demoparser2"
            ),
            "sample_ticks": sample_ticks,
            "dropped_duplicate_game_time_samples": dropped_time_samples,
            "round_outcome_reference": "Exact recorded round_end tick, inclusive; post-round deaths excluded",
            "eye_height": "Assumed 64 - 18*duck_amount units, not measured camera telemetry",
            "identity": "Exact SteamID64; owner/entity identifiers come from smoke events, not spatial matching",
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".building")
    opener = gzip.open if output.suffix == ".gz" else open
    with opener(temporary, "wt", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    temporary.replace(output)
    report = {
        "map": expected_map,
        "rounds": len(rounds),
        "players": len(roster),
        "smokes": sum(len(r["events"]["smokes"]) for r in rounds),
        "source": data["source"],
        "export_sha256": sha256_file(output),
    }
    output.with_suffix(output.suffix + ".metadata.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    return report
