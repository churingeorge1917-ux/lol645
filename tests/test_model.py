import contextlib
import copy
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import joblib
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from scope_cache import ROOT
from scope_cache.core import metrics, split_indices
from scope_cache.data import RELEASE_FEATURES, previous_activity
from scope_cache.model import (
    SETUP_FEATURES,
    setup_input,
    balanced_threshold,
    inputs,
    load,
    matrix,
    predict,
    predict_setup,
    precision_threshold,
    train,
)
from scope_cache.validation import WeightedMetrics, cluster_intervals


def grenade(pid, tick, side=2, round_index=0, player="a"):
    return {
        "GeneralInfo": {
            "ProjectileID": pid,
            "StartTick": tick,
            "Team": side,
            "RoundIndexMatch": round_index,
            "ThrowerID": player,
        },
        "FlashedPlayers": [],
    }


class TemporalContextTests(unittest.TestCase):
    def test_same_tick_and_future_releases_are_excluded(self):
        observations = [
            ("Smokes", grenade("past", 10, 3)),
            ("Flashbangs", grenade("one", 20)),
            ("Smokes", grenade("tie", 20)),
            ("HEs", grenade("future", 30)),
        ]
        result = previous_activity(observations)[(0, "one")]
        self.assertEqual(result["past_smokes_opponent"], 1)
        self.assertEqual(result["past_smokes_own"], 0)
        self.assertEqual(result["past_hes_own"], 0)
        self.assertEqual(result["ticks_since_opponent_grenade"], 10)
        self.assertEqual(result["ticks_since_own_grenade"], -1)

    def test_future_effects_and_order_cannot_change_present_features(self):
        observations = [
            ("Smokes", grenade("past", 10, 3)),
            ("Flashbangs", grenade("one", 20)),
            ("HEs", grenade("future", 30)),
        ]
        expected = previous_activity(observations)[(0, "one")]
        changed = copy.deepcopy(observations)
        changed[-1][1]["FlashedPlayers"] = [{"FlashDurationAdded": 999}]
        changed[-1][1]["GeneralInfo"]["PosTo"] = {"X": 1e9, "Y": 1e9, "Z": 1e9}
        self.assertEqual(previous_activity(list(reversed(changed)))[(0, "one")], expected)

    def test_round_boundaries_reset_activity(self):
        values = previous_activity(
            [
                ("Smokes", grenade("old", 10)),
                ("Flashbangs", grenade("new", 100, round_index=1)),
            ]
        )[(1, "new")]
        self.assertEqual(values["past_smokes_own"], 0)
        self.assertEqual(values["ticks_since_first_grenade"], 0)


class StatisticalContractTests(unittest.TestCase):
    def test_weighted_tied_scores_match_sklearn(self):
        from sklearn.metrics import (
            average_precision_score,
            roc_auc_score,
            accuracy_score,
            balanced_accuracy_score,
            brier_score_loss,
            log_loss,
        )

        y = np.array([0, 1, 1, 0, 0, 1])
        p = np.array([0.4, 0.4, 0.8, 0.8, 0.1, 0.9])
        w = np.array([2, 1, 0, 3, 4, 1])
        actual = WeightedMetrics(y, p, 0.5).evaluate(w)
        expected = {
            "roc_auc": roc_auc_score(y, p, sample_weight=w),
            "average_precision": average_precision_score(y, p, sample_weight=w),
            "accuracy": accuracy_score(y, p >= 0.5, sample_weight=w),
            "balanced_accuracy": balanced_accuracy_score(y, p >= 0.5, sample_weight=w),
            "brier": brier_score_loss(y, p, sample_weight=w),
            "log_loss": log_loss(y, p, sample_weight=w),
        }
        for key in expected:
            self.assertAlmostEqual(actual[key], expected[key], places=12)

    def test_cluster_bootstrap_repeats_matches_and_is_reproducible(self):
        y = np.array([0, 1] * 8)
        p = np.linspace(0.1, 0.9, 16)
        groups = np.repeat(np.arange(8), 2)
        one = cluster_intervals(y, p, groups, repetitions=30)
        two = cluster_intervals(y, p, groups, repetitions=30)
        self.assertEqual(one, two)
        self.assertEqual(one["clusters"], 8)
        self.assertAlmostEqual(one["metrics"]["roc_auc"]["estimate"], metrics(y, p)["roc_auc"])

    def test_precision_policy_does_not_invent_an_available_operating_point(self):
        self.assertIsNone(precision_threshold(np.zeros(1000), np.ones(1000) * 0.7))
        y = np.r_[np.ones(800), np.zeros(200)]
        p = np.r_[np.ones(800) * 0.9, np.ones(200) * 0.2]
        self.assertEqual(precision_threshold(y, p), 0.2)
        self.assertTrue(0 <= balanced_threshold(y, p) <= 1)


class CurrentInferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle, cls.meta = load()
        cls.row = json.loads((ROOT / "tests/fixtures/grenades.json").read_text())[1]

    def test_future_outcomes_cannot_change_release_prediction(self):
        changed = dict(
            self.row,
            impact=1 - self.row["impact"],
            enemy_flash_added=999,
            enemy_blinded_players=5,
            end_x=-9999,
        )
        np.testing.assert_array_equal(inputs([self.row]), inputs([changed]))
        self.assertEqual(
            predict([self.row], self.bundle, self.meta),
            predict([changed], self.bundle, self.meta),
        )

    def test_release_context_is_required_and_setup_is_explicit(self):
        row = {key: value for key, value in self.row.items() if key not in RELEASE_FEATURES}
        with self.assertRaisesRegex(ValueError, "missing release feature"):
            inputs([row])
        self.assertEqual(predict_setup([row], self.bundle, self.meta)[0]["mode"], "setup")
        for change in [
            {"past_player_throws": -0.1},
            {"ticks_since_first_grenade": -1},
            {"past_smokes_own": 0.5},
            {"release_offset_x": float("inf")},
        ]:
            with self.subTest(change=change), self.assertRaises(ValueError):
                inputs([dict(self.row, **change)])

    def test_count_prediction_has_physical_support(self):
        value = predict([self.row], self.bundle, self.meta)[0]
        self.assertGreaterEqual(value["conditional_blinded_enemy_players"], 1)
        self.assertLessEqual(value["conditional_blinded_enemy_players"], 5)
        self.assertLessEqual(value["expected_blinded_enemy_players"], 5)
        self.assertIsNone(value["tactical_success"])


class TrainingBoundaryTests(unittest.TestCase):
    def test_holdout_labels_cannot_change_selection_threshold_or_evaluation_fit(self):
        rows = []
        rng = np.random.default_rng(812)
        for group in range(30):
            for i in range(20):
                label = i % 2
                row = {key: float(rng.normal()) for key in SETUP_FEATURES}
                row.update({key: 0.0 for key in RELEASE_FEATURES})
                row.update(
                    type="Flashbangs",
                    match_id=str(group),
                    player_id="p",
                    projectile_id=str(i),
                    round_index=i,
                    side=2 + group % 2,
                    source_row_index=len(rows),
                    impact=label,
                    enemy_blinded_players=label,
                    enemy_flash_added=float(label),
                )
                rows.append(row)
        original = pa.Table.from_pylist(rows)
        x = matrix(original)
        y = original["impact"].to_numpy()
        groups = original["match_id"].to_numpy()
        _, test, _, _ = split_indices(x, y, groups, 42, 3)
        altered = copy.deepcopy(rows)
        for i in test:
            value = 1 - altered[i]["impact"]
            altered[i].update(
                impact=value,
                enemy_blinded_players=value,
                enemy_flash_added=float(value),
            )
        prototypes = {
            "setup_catboost": (
                "setup",
                make_pipeline(StandardScaler(), LogisticRegression(max_iter=300, random_state=42)),
            ),
            "release_hist": (
                "release",
                DecisionTreeClassifier(max_depth=3, min_samples_leaf=20, random_state=42),
            ),
        }
        reports = []
        fits = []
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch("scope_cache.model.candidates", return_value=prototypes),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            for n, data in enumerate([rows, altered]):
                folder = Path(tmp) / str(n)
                folder.mkdir()
                events = folder / "events.parquet"
                context = folder / "context.parquet"
                table = pa.Table.from_pylist(data)
                pq.write_table(table, events)
                context.write_bytes(b"fixture-context")
                with patch("scope_cache.model.flash_table", return_value=table):
                    output = folder / "model.joblib"
                    reports.append(train(events, context, output, folds=3, include_kernel=False))
                    bundle = joblib.load(output)
                    fits.append(bundle["evaluation"]["classification"])
        first, second = reports
        for key in [
            "selected_model",
            "decision_threshold",
            "high_precision_threshold",
            "selection",
        ]:
            self.assertEqual(first[key], second[key])
        profile = first["classification_profile"]
        xx = x if profile == "release" else x[:, : len(SETUP_FEATURES)]
        np.testing.assert_allclose(
            fits[0].predict_proba(xx), fits[1].predict_proba(xx), rtol=0, atol=1e-12
        )


class SetupInferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle, cls.metadata = load()
        cls.row = json.loads((ROOT / "tests/fixtures/grenades.json").read_text())[1]

    def test_prethrow_features_exclude_recorded_effect(self):
        first = setup_input([self.row])
        altered = dict(self.row, impact=0, enemy_flash_added=999, end_x=999999)
        np.testing.assert_array_equal(first, setup_input([altered]))
        self.assertFalse(
            any("end_" in f or "damage" in f or "flash_added" in f for f in SETUP_FEATURES)
        )

    def test_invalid_inputs_and_probability_severity(self):
        for change in [
            {"view_dx": float("nan")},
            {"view_dx": 4},
            {"side": 2.5},
            {"round_index": -1},
            {"is_airborne": 4},
            {"type": "Smokes"},
        ]:
            with self.subTest(change=change), self.assertRaises(ValueError):
                setup_input([dict(self.row, **change)])
        with self.assertRaises(ValueError):
            setup_input([{k: v for k, v in self.row.items() if k != "velocity_x"}])
        result = predict_setup([self.row], self.bundle, self.metadata)[0]
        self.assertGreaterEqual(result["registered_enemy_effect_probability"], 0)
        self.assertLessEqual(result["registered_enemy_effect_probability"], 1)
        self.assertGreaterEqual(result["expected_registered_flash_added"], 0)

    def test_checksum_is_checked_before_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.joblib"
            path.write_bytes(b"bad model")
            path.with_suffix(".metadata.json").write_text(json.dumps(self.metadata))
            with self.assertRaisesRegex(ValueError, "checksum"):
                load(path)
