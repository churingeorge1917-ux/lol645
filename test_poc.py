"""Проверки геометрии, профилей, разбиения ML и целостности поставляемой базы."""

import csv
import pickle
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent
import ml_algorithm as model


def sample_match():
    """Маленький синтетический матч; не часть эмпирических результатов."""

    def point(x, y, z):
        return {"X": x, "Y": y, "Z": z}

    def grenade(identifier, thrower=11):
        return {
            "GeneralInfo": {
                "RoundIndexDemo": 0,
                "RoundIndexMatch": 0,
                "ProjectileID": identifier,
                "ThrowerID": thrower,
                "Team": 2,
                "StartTick": 0,
                "EndTick": 64,
                "StartFrame": 0,
                "EndFrame": 64,
                "PosFrom": point(860.0, -872.0, 1678.0),
                "PosTo": point(10.0, 20.0, 1700.0),
                "ThrowPoint": {
                    "Position": point(879.802185, -872.03125, 1614.03125),
                    "Velocity": point(0.0, 0.0, 0.0),
                    "ViewX": 177.892853,
                    "ViewY": -14.157107,
                    "IsCrouching": False,
                    "IsAirborne": False,
                },
            }
        }

    flash, he = grenade(1), grenade(2)
    flash["FlashedPlayers"] = [
        {
            "PlayerID": 222,
            "Team": 3,
            "FlashDurationAdded": 0.25,
            "Position": point(10.0, 20.0, 1700.0),
        }
    ]
    he["DamagedPlayers"] = [
        {
            "PlayerID": 222,
            "Team": 3,
            "Damage": [
                {
                    "HealthDamage": 1,
                    "Position": point(10.0, 20.0, 1700.0),
                    "TickNumber": 65,
                }
            ],
        }
    ]
    return {
        "MatchID": 1,
        "Players": [{"StaticInfo": {"PlayerID": p}} for p in (11, 12, 13)],
        "Flashbangs": [flash],
        "HEs": [he],
        "Fires": [grenade(3)],
        "Smokes": [grenade(4, 12)],
    }


class ScopeTests(unittest.TestCase):
    def row(self, impact=1, **changes):
        row = {
            "match_id": "m",
            "player_id": "p",
            "round": 0,
            "projectile_id": "g",
            "side": 2,
            "type": "Flashbangs",
            "reference": "r",
            "standing": 1,
            "speed": 0.0,
            "position_error_pct": 1.0,
            "direction_error_pct": 1.0,
            "landing_error_pct": None,
            "impact": impact,
            "player_x": 1.0,
            "player_y": 2.0,
            "player_z": 3.0,
        }
        row.update(changes)
        return row

    def test_full_profiles_use_current_impact_definition(self):
        rows = {r["player_id"]: r for r in model.match_profiles(sample_match())}
        player = rows["11"]
        self.assertEqual(player["total_grenades"], 3)
        self.assertEqual(player["positive_outcomes"], 2)
        self.assertEqual(player["known_outcomes"], 2)
        self.assertEqual(player["unknown_outcomes"], 1)
        self.assertEqual(player["enemy_damage"], 1.0)
        self.assertEqual(player["enemy_flash_added"], 0.25)
        self.assertEqual(player["observed_impact_rate"], 1.0)
        self.assertAlmostEqual(player["effect_coverage"], 2 / 3)
        self.assertAlmostEqual(player["impact_lower_bound"], 2 / 3)
        self.assertEqual(player["impact_upper_bound"], 1.0)
        self.assertEqual(rows["12"]["total_grenades"], 1)
        self.assertEqual(rows["12"]["smokes"], 1)
        self.assertIsNone(rows["12"]["observed_impact_rate"])
        self.assertIsNone(rows["13"]["effect_coverage"])
        self.assertEqual(rows["13"]["total_grenades"], 0)

    def test_full_profiles_reject_duplicate_projectiles(self):
        fixture = sample_match()
        fixture["Fires"][0]["GeneralInfo"]["ProjectileID"] = 1
        with self.assertRaises(ValueError):
            model.match_profiles(fixture)

    def test_known_zero_effect_is_not_unknown(self):
        fixture = sample_match()
        fixture["Flashbangs"][0]["FlashedPlayers"][0]["FlashDurationAdded"] = 0.0
        player = next(
            r for r in model.match_profiles(fixture) if r["player_id"] == "11"
        )
        self.assertEqual(player["zero_effect_outcomes"], 1)
        self.assertEqual(player["unknown_outcomes"], 1)
        self.assertEqual(player["positive_outcomes"], 1)

    def test_unknown_endpoint_stays_unknown(self):
        row = self.row()
        self.assertIsNone(model.full_matches(row, 15))
        self.assertIsNone(model.summarize_geometry([row])[0]["full"])

    def test_missing_effect_is_not_zero(self):
        effect, _ = model.enemy_effect({"GeneralInfo": {"Team": 2}}, "Flashbangs")
        self.assertIsNone(effect["impact"])
        aggregate = model.player_rounds([self.row(None)])
        self.assertIsNone(next(iter(aggregate.values()))["utility_rate"])

    def test_negative_grid_and_angle_wrap(self):
        self.assertEqual(model.cell((-1.0, -256.0, -65.0)), (-1, -1, -2))
        _, degrees = model.direction_error(0, 179, {"pitch": 0, "yaw": -179})
        self.assertAlmostEqual(degrees, 2.0)

    def test_phi_and_unknown_denominator(self):
        rows = [
            self.row(1, match_id="1"),
            self.row(0, match_id="2"),
            self.row(0, position_error_pct=20, match_id="3"),
            self.row(None, position_error_pct=20, match_id="4"),
        ]
        result = model.reference_associations(rows, 10)[0]
        self.assertEqual(result["within_known"], 2)
        self.assertEqual(result["outside_total"], 2)
        self.assertEqual(result["outside_known"], 1)
        self.assertAlmostEqual(result["phi"], 0.5)
        self.assertIsNone(result["delta_low"])

    def test_flash_predictors_do_not_include_effects(self):
        rows = [self.row(enemy_flash_added=10, end_x=100), self.row(None)]
        with patch.object(
            model, "fit_classifier", return_value=({"status": "test"}, [])
        ) as fit:
            model.flash_model(rows)
        features, labels, matches = fit.call_args.args
        self.assertEqual(len(features), 1)
        self.assertEqual(labels, [1])
        self.assertEqual(matches, ["m"])
        self.assertFalse({"enemy_flash_added", "end_x", "impact"} & set(features[0]))

    def test_group_split_and_train_only_categories(self):
        features = [{"x": float(i % 2), "reference": str(i // 2)} for i in range(80)]
        labels = [i % 2 for i in range(80)]
        groups = [str(i // 2) for i in range(80)]
        result, predictions = model.fit_classifier(features, labels, groups)
        self.assertEqual(result["status"], "trained")
        self.assertFalse(set(result["train_match_ids"]) & set(result["test_match_ids"]))
        self.assertEqual(len(predictions), 80)
        for group in result["test_match_ids"]:
            self.assertNotIn("reference=" + group, result["features"])

    def test_constant_labels_are_not_fitted(self):
        result, predictions = model.fit_classifier(
            [{"x": i} for i in range(20)], [1] * 20, list(map(str, range(20)))
        )
        self.assertEqual(result["status"], "not_estimable")
        self.assertFalse(predictions)

    def context_file(self, rows, directory):
        path = Path(directory) / "context.csv"
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(model.CONTEXT_COLUMNS)
            writer.writerows(rows)
        return path

    def test_context_rejects_invalid_joins(self):
        good = ("m", 0, "p", 2, 1, 0.5)
        cases = [
            [good, good],
            [good, ("m", 0, "q", 2, 0, 0.4)],
            [good, ("m", 0, "q", 3, 1, 0.4)],
            [good, ("m", 0, "p", 3, 0, 0.4)],
            [("m", 0, "p", 2, 1, "nan")],
            [("unknown", 0, "p", 2, 1, 0.5)],
        ]
        with tempfile.TemporaryDirectory() as directory:
            for rows in cases:
                with self.subTest(rows=rows), self.assertRaises(ValueError):
                    model.load_context(
                        self.context_file(rows, directory),
                        "Independent score",
                        {"m"},
                    )

    def test_optional_round_model_grain(self):
        rows, context = [], []
        for match in range(20):
            for side in (2, 3):
                for player in range(5):
                    player_id = f"{side}/{player}"
                    won = (match + side) % 2
                    context.append(
                        {
                            "match_id": str(match),
                            "round_index": 0,
                            "player_id": player_id,
                            "side": side,
                            "round_won": won,
                            "player_kpd": float(player + match % 3),
                        }
                    )
                    rows.append(
                        self.row(
                            int(player % 2 == 0),
                            match_id=str(match),
                            side=side,
                            player_id=player_id,
                        )
                    )
        result, predictions = model.outcome_analysis(
            rows, context, "synthetic validation only", 20
        )
        self.assertEqual(result["joined_player_rounds"], 200)
        self.assertEqual(result["round_model"]["status"], "trained")
        self.assertEqual(len(predictions), 40)
        self.assertEqual(
            len({(p["match_id"], p["round_index"], p["side"]) for p in predictions}), 40
        )
        self.assertAlmostEqual(result["correlations"][1]["pearson"], 0.0)
        constant = [{"match_id": str(i), "x": 1.0, "y": i % 2} for i in range(10)]
        self.assertIsNone(model.correlation_interval(constant, "x", "y", 20)["pearson"])

    def test_full_dataset_audit(self):
        with model.database(model.DEFAULT_DATABASE) as connection:
            self.assertEqual(
                connection.execute("PRAGMA integrity_check").fetchone()[0], "ok"
            )
            self.assertEqual(
                tuple(
                    connection.execute(
                        "SELECT count(*),count(distinct match_id) FROM throws"
                    ).fetchone()
                ),
                (167499, 9876),
            )
            totals = connection.execute(
                "SELECT point,sum(throws) FROM spatial_grid GROUP BY point"
            ).fetchall()
            self.assertEqual(
                {r[0]: r[1] for r in totals}, {"player": 2385417, "end": 2385417}
            )
            profile = model.read_document(connection, "player_profiles")
            self.assertEqual(profile["matches"], 10000)
            self.assertEqual(profile["player_matches"], 99921)
            self.assertEqual(profile["throws"], 2385417)
            self.assertEqual(profile["known"], 795325)
            self.assertEqual(profile["positive"], 599578)
            self.assertEqual(
                connection.execute("SELECT COUNT(*) FROM player_matches").fetchone()[0],
                99921,
            )
            self.assertEqual(
                connection.execute(
                    "SELECT COUNT(*) FROM player_matches WHERE known_outcomes != positive_outcomes + zero_effect_outcomes OR eligible_grenades != known_outcomes + unknown_outcomes OR total_grenades != smokes + flashbangs + fires + hes"
                ).fetchone()[0],
                0,
            )
            model_result = model.read_document(connection, "flash_model")
            self.assertFalse(
                set(model_result["train_match_ids"])
                & set(model_result["test_match_ids"])
            )
            predictions = model.read_document(connection, "flash_predictions")
            test = [r for r in predictions if r["partition"] == "test"]
            y = model.np.array([r["label"] for r in test])
            probabilities = model.np.array([r["probability"] for r in test])
            self.assertAlmostEqual(
                float(model.np.mean((y - probabilities) ** 2)),
                model_result["test"]["metrics"]["brier"],
            )
            self.assertEqual(
                model.read_document(connection, "outcomes")["status"], "missing_context"
            )

    def test_preparation_on_one_fixture(self):
        example = sample_match()
        expected = sum(
            len(example[k]) for k in ("Smokes", "Flashbangs", "Fires", "HEs")
        )
        with tempfile.TemporaryDirectory() as directory:
            target, source = (
                Path(directory) / "fixture.sqlite",
                Path(directory) / "fixture.list",
            )
            with (
                sqlite3.connect(model.DEFAULT_DATABASE) as original,
                sqlite3.connect(target) as new,
            ):
                for row in original.execute(
                    "SELECT sql FROM sqlite_master WHERE type='table'"
                ):
                    new.execute(row[0])
                new.execute(
                    "INSERT INTO documents VALUES (?,?)",
                    original.execute(
                        "SELECT * FROM documents WHERE key='references'"
                    ).fetchone(),
                )
            original.close()
            new.close()
            source.write_bytes(pickle.dumps([example], protocol=4))
            model.rebuild_database(source, target)
            with model.database(target) as connection:
                extraction = model.read_document(connection, "extraction")
                self.assertEqual(extraction["matches"], 1)
                self.assertEqual(extraction["counts"]["throws"], expected)
                grid = connection.execute(
                    "SELECT sum(throws) FROM spatial_grid WHERE point='player'"
                ).fetchone()[0]
                self.assertEqual(grid, expected)
            summary = model.refresh_player_matches(source, target)
            self.assertEqual(summary["matches"], 1)
            self.assertEqual(summary["player_matches"], 3)
            self.assertEqual(summary["throws"], expected)
            with model.database(target) as connection:
                self.assertEqual(
                    connection.execute(
                        "SELECT COUNT(*) FROM player_matches"
                    ).fetchone()[0],
                    3,
                )
                self.assertLess(
                    connection.execute("SELECT COUNT(*) FROM throws").fetchone()[0],
                    expected,
                )
            before = target.read_bytes()
            source.write_bytes(b"incorrect source")
            with self.assertRaises(ValueError):
                model.refresh_player_matches(source, target)
            self.assertEqual(target.read_bytes(), before)
            source.write_bytes(pickle.dumps(Exception("unsupported")))
            with self.assertRaises(pickle.UnpicklingError):
                model.rebuild_database(source, target)
            self.assertEqual(target.read_bytes(), before)
            self.assertFalse(target.with_suffix(".building.sqlite").exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
