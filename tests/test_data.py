import pickle
import tempfile
import unittest
from pathlib import Path
from scope_cache.core import read_matches, enemy_effect
from scope_cache.data import blind_counts


class SourceTests(unittest.TestCase):
    def test_stream_reader_preserves_exact_match_ids(self):
        rows = [
            {"MatchID": 3831383152580887235 + i, "Players": [], "data": [1, 2, 3]}
            for i in range(10)
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "source.list"
            path.write_bytes(pickle.dumps(rows, protocol=4))
            result = []
            read_matches(path, lambda row: result.append(row["MatchID"]))
            self.assertEqual(result, [r["MatchID"] for r in rows])

    def test_unpickler_rejects_unexpected_globals(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "source.list"
            path.write_bytes(pickle.dumps(Path("untrusted"), protocol=4))
            with self.assertRaises(pickle.UnpicklingError):
                read_matches(path, lambda row: None)

    def test_missing_outcome_is_distinct_from_known_zero(self):
        grenade = {"GeneralInfo": {"Team": 2, "ThrowerID": "thrower"}}
        self.assertIsNone(enemy_effect(grenade, "Flashbangs")[0]["impact"])
        grenade["FlashedPlayers"] = []
        self.assertEqual(enemy_effect(grenade, "Flashbangs")[0]["impact"], 0)

    def test_blind_targets_are_deduplicated_and_team_damage_is_separate(self):
        def target(pid, team, value):
            return {
                "PlayerID": pid,
                "Team": team,
                "FlashDurationAdded": value,
                "Position": {"X": 0, "Y": 0, "Z": 0},
            }

        grenade = {
            "GeneralInfo": {"Team": 2, "ThrowerID": "self"},
            "FlashedPlayers": [
                target("enemy", 3, 2),
                target("enemy", 3, 3),
                target("ally", 2, 1),
                target("self", 2, 4),
            ],
        }
        effect, _ = enemy_effect(grenade, "Flashbangs")
        self.assertEqual(effect["enemy_flash_added"], 3)
        self.assertEqual(effect["ally_flash_added"], 1)
        self.assertEqual(effect["self_flash_added"], 4)
        self.assertEqual(blind_counts(grenade, "Flashbangs")["enemy_blinded_players"], 1)


class ObservedEffectTests(unittest.TestCase):
    def test_nonapplicable_effects_are_not_known_zero(self):
        import pyarrow as pa
        from scope_cache.scoring import observed_columns

        types = ["Smokes", "Flashbangs", "HEs", "Fires", "HEs", "Flashbangs"]
        impacts = [None, 1, 1, 1, None, 0]
        columns = {"type": types, "impact": impacts}
        for side in ["enemy", "ally", "self"]:
            columns[side + "_damage"] = [0.0, 0.0, 10.0, 30.0, 0.0, 0.0]
            columns[side + "_flash_added"] = [0.0, 5.0, 0.0, 0.0, 0.0, 0.0]
        result = observed_columns(pa.table(columns))
        self.assertEqual(
            result["observed_enemy_damage"].to_pylist(), [None, None, 10.0, 30.0, None, None]
        )
        self.assertEqual(
            result["observed_enemy_flash_added"].to_pylist(), [None, 5.0, None, None, None, 0.0]
        )
        self.assertEqual(
            result["outcome_status"].to_pylist(),
            [
                "not_applicable_smoke",
                "known_positive",
                "known_positive",
                "known_positive",
                "unknown",
                "known_zero",
            ],
        )
