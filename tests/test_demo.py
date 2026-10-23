import copy
import json
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import pyarrow as pa
import pyarrow.parquet as pq
from scope_cache import ROOT
from scope_cache.demo import (
    account32,
    import_demo,
    load_demo,
    round_records,
    steamid,
    ROUND_SCHEMA,
    join_events,
    smoke_exposure,
)
from scope_cache.geometry import (
    MapGeometry,
)


def demo_fixture():
    def player(x, yaw):
        return {
            "x": [x] * 3,
            "y": [0.0] * 3,
            "z": [0.0] * 3,
            "yaw": [yaw] * 3,
            "pitch": [0.0] * 3,
            "alive": [True] * 3,
        }

    return {
        "format": "cs2.demo.v2",
        "map": "de_cache",
        "players": [{"steamid": "one"}, {"steamid": "two"}],
        "weapons": {"ak47": 0, "hegrenade": 1},
        "rounds": [
            {
                "id": 0,
                "teams": ["T", "CT"],
                "winner": "T",
                "end_reason": "ct_killed",
                "ticks": [100, 116, 132],
                "round_seconds": [0.0, 0.25, 0.5],
                "players": [player(-200, 0), player(200, 180)],
                # Death occurs AFTER the last sampled state.
                "events": {
                    "kills": [{"t": 135, "a": 0, "v": 1}],
                    "damage": [
                        {"t": 130, "a": 0, "v": 1, "w": 0, "hp": 75},
                        {"t": 132, "a": 0, "v": 1, "w": 1, "hp": 25},
                    ],
                    "smokes": [{"ts": 0.0, "te": 0.5, "x": 0, "y": 0, "z": 0}],
                },
            }
        ],
    }


class CSNetAdapterTests(unittest.TestCase):
    def test_smoke_import_runs_with_the_real_cache_geometry(self):
        demo = demo_fixture()
        geometry = MapGeometry()
        a, b = geometry.a[0], geometry.b[0]
        for player, eye, target in zip(demo["rounds"][0]["players"], [a, b], [b, a]):
            for axis, value in zip("xyz", eye - [0, 0, 64]):
                player[axis] = [float(value)] * 3
            dx, dy, dz = target - eye
            player["yaw"] = [math.degrees(math.atan2(dy, dx))] * 3
            player["pitch"] = [-math.degrees(math.atan2(dz, math.hypot(dx, dy)))] * 3
        center = (a + b) / 2 - [0, 0, 72]
        demo["rounds"][0]["events"]["smokes"][0].update(dict(zip("xyz", map(float, center))))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "synthetic.json"
            path.write_text(json.dumps(demo))
            output = Path(tmp) / "out"
            import_demo(path, output, match_id="synthetic-not-research-data", score_smokes=True)
            scores = json.loads((output / "smoke_exposure.json").read_text())
            self.assertEqual(scores[0]["blocked_pair_seconds_T"], 0.5)
            self.assertEqual(scores[0]["marginal_blocked_pair_seconds_CT"], 0.5)

    def test_survival_uses_deaths_after_last_sample_and_explicit_round_offset(self):
        demo = demo_fixture()
        rows = round_records(demo, demo["rounds"][0], "m", round_offset=1)
        self.assertEqual([r["alive_end"] for r in rows], [1, 0])
        self.assertEqual([r["round_won"] for r in rows], [1, 0])
        self.assertEqual(rows[0]["weapon_enemy_damage"], 75)
        self.assertEqual(rows[0]["enemy_damage"], 100)
        self.assertEqual(rows[0]["round_index"], 1)

    def test_missing_labels_never_default_to_survival_or_ct_win(self):
        demo = demo_fixture()
        data = demo["rounds"][0]
        del data["events"]["kills"]
        del data["events"]["damage"]
        data["winner"] = ""
        rows = round_records(demo, data, "m")
        for row in rows:
            for key in ["alive_end", "round_won", "enemy_kills", "enemy_damage"]:
                self.assertIsNone(row[key])

    def test_smoke_integration_and_marginal_overlap(self):
        data = demo_fixture()["rounds"][0]
        geometry = SimpleNamespace(visible=lambda a, b: True)
        score = smoke_exposure(data, geometry)[0]
        self.assertEqual(score["blocked_pair_seconds_T"], 0.5)
        self.assertEqual(score["blocked_pair_seconds_CT"], 0.5)
        self.assertEqual(score["marginal_blocked_pair_seconds_T"], 0.5)
        data["events"]["smokes"].append(copy.deepcopy(data["events"]["smokes"][0]))
        for score in smoke_exposure(data, geometry):
            self.assertEqual(score["blocked_pair_seconds_T"], 0.5)
            self.assertEqual(score["marginal_blocked_pair_seconds_T"], 0.0)
        geometry.visible = lambda a, b: False
        self.assertEqual(smoke_exposure(data, geometry)[0]["blocked_pair_seconds_T"], 0.0)

    def test_smoke_boundary_clipping_gaps_and_view_direction(self):
        data = demo_fixture()["rounds"][0]
        data["events"]["smokes"][0].update(ts=0.1, te=0.3)
        geometry = SimpleNamespace(visible=lambda a, b: True)
        self.assertAlmostEqual(smoke_exposure(data, geometry)[0]["blocked_pair_seconds_T"], 0.2)
        self.assertEqual(
            smoke_exposure(data, geometry, max_gap=0.1)[0]["blocked_pair_seconds_T"], 0
        )
        data["players"][0]["yaw"] = [180] * 3
        self.assertEqual(smoke_exposure(data, geometry)[0]["blocked_pair_seconds_T"], 0)

    def test_import_rejects_other_maps_and_exact_joins_preserve_unknowns(self):
        demo = demo_fixture()
        rows = round_records(demo, demo["rounds"][0], "m")
        context = pa.Table.from_pylist(rows, schema=ROUND_SCHEMA)
        events = pa.Table.from_pylist(
            [
                {
                    "match_id": "m",
                    "round_index": 0,
                    "player_id": "one",
                    "side": 2,
                    "impact": None,
                },
                {
                    "match_id": "different",
                    "round_index": 0,
                    "player_id": "one",
                    "side": 2,
                    "impact": 1,
                },
                {
                    "match_id": "m",
                    "round_index": 1,
                    "player_id": "one",
                    "side": 2,
                    "impact": 0,
                },
            ]
        )
        joined = join_events(events, context)
        self.assertEqual(joined["alive_end"].to_pylist(), [1, None, None])
        self.assertEqual(joined["impact"].to_pylist(), [None, 1, 0])
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            join_events(events, pa.concat_tables([context, context]))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "demo.json"
            path.write_text(json.dumps(demo))
            output = Path(tmp) / "out"
            report = import_demo(path, output, match_id="m", kpd="weapon-enemy-damage")
            self.assertEqual(report["alive_end_known"], 2)
            self.assertEqual(pq.read_table(output / "player_states.parquet").num_rows, 6)
            demo["map"] = "de_mirage"
            path.write_text(json.dumps(demo))
            with self.assertRaisesRegex(ValueError, "Map mismatch"):
                import_demo(path, Path(tmp) / "other", match_id="m")


class DemoIntegrityTests(unittest.TestCase):
    def test_exact_steam_identity_and_float_rejection(self):
        self.assertEqual(account32("76561197960265851"), "123")
        self.assertEqual(steamid("76561197960265851"), "76561197960265851")
        with self.assertRaises(ValueError):
            steamid(float(76561197960265851))
        with self.assertRaises(ValueError):
            account32(float(76561197960265851))
        fixture = load_demo(ROOT / "tests/fixtures/demoparser_round.json.gz")
        fixture["players"][0]["steamid"] = float(76561197960265851)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "lossy.json"
            path.write_text(json.dumps(fixture))
            with self.assertRaisesRegex(ValueError, "float IDs"):
                load_demo(path)

    def test_recorded_smoke_owner_is_retained_on_real_parser_fixture(self):
        fixture = ROOT / "tests/fixtures/demoparser_round.json.gz"
        data = load_demo(fixture)
        expected = sum(
            s.get("thrower_id") is not None for rd in data["rounds"] for s in rd["events"]["smokes"]
        )
        with tempfile.TemporaryDirectory() as tmp:
            report = import_demo(
                fixture,
                Path(tmp) / "output",
                match_id="fixture-only",
                expected_map="de_mirage",
                id_scheme="steam-account32",
            )
            self.assertEqual(report["smokes_with_recorded_owner"], expected)
            intervals = pq.read_table(Path(tmp) / "output/smoke_intervals.parquet")
            self.assertTrue(all(v is not None for v in intervals["projectile_id"].to_pylist()))

    def test_failed_import_is_atomic_and_map_mismatch_is_rejected(self):
        fixture = ROOT / "tests/fixtures/demoparser_round.json.gz"
        data = load_demo(fixture)
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "out"
            with self.assertRaisesRegex(ValueError, "Map mismatch"):
                import_demo(fixture, output, match_id="fixture")
            self.assertFalse(output.exists())
            incompatible = copy.deepcopy(data)
            incompatible["map"] = "de_cache"
            incompatible.setdefault("source", {}).setdefault("header", {})["patch_version"] = "0"
            incompatible_path = Path(tmp) / "incompatible.json"
            incompatible_path.write_text(json.dumps(incompatible))
            with self.assertRaisesRegex(ValueError, "differs from geometry patch"):
                import_demo(incompatible_path, output, match_id="fixture", score_smokes=True)
            self.assertFalse(output.exists())
            data["rounds"][0]["players"][0]["x"] = []
            path = Path(tmp) / "bad.json"
            path.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, "Unaligned"):
                import_demo(path, output, match_id="fixture", expected_map="de_mirage")
            self.assertFalse(output.exists())

    def test_explicit_round_end_excludes_postround_death(self):
        d = demo_fixture()
        rd = d["rounds"][0]
        rd["end_tick"] = 134
        result = round_records(d, rd, "m")
        self.assertEqual(result[1]["alive_end"], 1)
        rd["end_tick"] = 140
        self.assertEqual(round_records(d, rd, "m")[1]["alive_end"], 0)
