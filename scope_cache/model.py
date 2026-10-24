"""Модель ослепления: выбор по матчам, проверка и прогноз."""

from __future__ import annotations
import json
import math
from pathlib import Path
import joblib
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from catboost import CatBoostClassifier
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, roc_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from threadpoolctl import threadpool_limits
from scope_cache import ROOT
from scope_cache.core import metrics, sha256_file, split_indices, fingerprint
from scope_cache.data import RELEASE_FEATURES

SETUP_FEATURES = [
    "player_x",
    "player_y",
    "player_z",
    "view_dx",
    "view_dy",
    "view_dz",
    "velocity_x",
    "velocity_y",
    "velocity_z",
    "is_crouching",
    "is_airborne",
    "side",
    "round_index",
]


def setup_input(rows):
    if not rows:
        raise ValueError("No flash observations supplied")
    result = []
    for index, row in enumerate(rows, 1):
        if row.get("type") != "Flashbangs":
            raise ValueError(f"Observation {index}: type must be Flashbangs")
        missing = [key for key in SETUP_FEATURES if key not in row or row[key] is None]
        if missing:
            raise ValueError(f"Observation {index}: missing {', '.join(missing)}")
        try:
            values = {key: float(row[key]) for key in SETUP_FEATURES}
        except (ValueError, TypeError) as error:
            raise ValueError(f"Observation {index}: features must be numeric") from error
        if not all(math.isfinite(v) for v in values.values()):
            raise ValueError(f"Observation {index}: features must be finite")
        if (
            values["side"] not in (2, 3)
            or values["is_crouching"] not in (0, 1)
            or values["is_airborne"] not in (0, 1)
        ):
            raise ValueError(f"Observation {index}: invalid side or movement state")
        if values["round_index"] < 0 or not values["round_index"].is_integer():
            raise ValueError(f"Observation {index}: round_index must be a nonnegative integer")
        norm = sum(values["view_d" + axis] ** 2 for axis in "xyz")
        if not math.isclose(norm, 1, abs_tol=1e-3):
            raise ValueError(f"Observation {index}: view direction must be a unit XYZ vector")
        result.append([values[key] for key in SETUP_FEATURES])
    if any(abs(value) > np.finfo(np.float32).max for row in result for value in row):
        raise ValueError("Input feature exceeds the supported numeric range")
    return np.asarray(result, dtype=np.float32)


DERIVED = [
    "movement_speed",
    "horizontal_speed",
    "view_velocity_dot",
    "release_distance",
    "view_release_cosine",
]


FEATURES = SETUP_FEATURES + RELEASE_FEATURES + DERIVED


VERSION = 4


DEFAULT_MODEL = ROOT / "models" / "flash.joblib"


def matrix(table, profile="release"):
    base = np.column_stack(
        [table[name].to_numpy(zero_copy_only=False) for name in SETUP_FEATURES]
    ).astype(np.float32)
    if profile == "setup":
        return base
    extra = np.column_stack(
        [table[name].to_numpy(zero_copy_only=False) for name in RELEASE_FEATURES]
    ).astype(np.float32)
    velocity = base[:, 6:9]
    view = base[:, 3:6]
    offset = extra[:, :3]
    speed = np.linalg.norm(velocity, axis=1)
    distance = np.linalg.norm(offset, axis=1)
    derived = np.column_stack(
        [
            speed,
            np.linalg.norm(velocity[:, :2], axis=1),
            np.sum(view * velocity, axis=1),
            distance,
            np.sum(view * offset, axis=1) / np.maximum(distance, 1e-9),
        ]
    )
    result = np.column_stack([base, extra, derived]).astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError(
            "Incomplete/nonfinite release features; use the setup model when release context is unavailable"
        )
    return result


def require_cache(rows):
    if any(row.get("map_name", "de_cache") != "de_cache" for row in rows):
        raise ValueError("The saved model is scoped to de_cache")


def inputs(rows):
    require_cache(rows)
    setup_input(rows)
    for i, row in enumerate(rows):
        for name in RELEASE_FEATURES:
            if name not in row or row[name] is None:
                raise ValueError(f"Row {i + 1}: missing release feature {name}")
            try:
                value = float(row[name])
            except (TypeError, ValueError) as error:
                raise ValueError(f"Invalid release feature {name}") from error
            if not math.isfinite(value) or abs(value) > np.finfo(np.float32).max:
                raise ValueError(f"Nonfinite release feature {name}")
            if name.startswith("past_") and (value < 0 or not value.is_integer()):
                raise ValueError(f"{name} must be a nonnegative count")
            if name.startswith("ticks_") and (
                value < (-1 if "first" not in name else 0) or not value.is_integer()
            ):
                raise ValueError(f"Invalid {name}")
    return matrix(
        pa.Table.from_pylist(
            [{key: float(row[key]) for key in SETUP_FEATURES + RELEASE_FEATURES} for row in rows]
        )
    )


def flash_table(
    events=ROOT / "data" / "events.parquet",
    context=ROOT / "data" / "release_context.parquet",
    known_only=True,
):
    flags = pq.read_table(events, columns=["type", "impact"])
    keep = pc.equal(flags["type"], "Flashbangs")
    if known_only:
        keep = pc.and_(keep, pc.is_valid(flags["impact"]))
    context_meta = json.loads(Path(context).with_suffix(".metadata.json").read_text())
    compact = context_meta.get("schema_version") == 2
    extra = (
        pq.read_table(context, filters=[("type", "=", "Flashbangs")])
        if compact
        else pq.read_table(context).filter(keep)
    )
    filters = [("type", "=", "Flashbangs")] + ([("impact", "in", [0, 1])] if known_only else [])
    data = pq.read_table(events, filters=filters)
    if compact and known_only:
        all_flash = pq.read_table(events, filters=[("type", "=", "Flashbangs")], columns=["impact"])
        extra = extra.filter(pc.is_valid(all_flash["impact"]))
    expected_indices = np.flatnonzero(keep.to_numpy(zero_copy_only=False))
    if len(extra) != len(data) or not np.array_equal(
        extra["source_row_index"].to_numpy(), expected_indices
    ):
        raise ValueError("Release context identity/order mismatch")
    for name in extra.column_names:
        if name not in data.column_names:
            data = data.append_column(name, extra[name])
    return data


def candidates():
    return {
        "setup_catboost": (
            "setup",
            CatBoostClassifier(
                iterations=250,
                depth=6,
                learning_rate=0.08,
                l2_leaf_reg=10,
                loss_function="Logloss",
                random_seed=42,
                thread_count=2,
                verbose=False,
                allow_writing_files=False,
            ),
        ),
        "release_logistic": (
            "release",
            make_pipeline(
                StandardScaler(),
                LogisticRegression(C=1, max_iter=700, tol=1e-3, random_state=42),
            ),
        ),
        "release_tree": (
            "release",
            DecisionTreeClassifier(max_depth=8, min_samples_leaf=300, random_state=42),
        ),
        "release_hist": (
            "release",
            HistGradientBoostingClassifier(
                max_iter=200,
                max_leaf_nodes=31,
                min_samples_leaf=100,
                l2_regularization=10,
                early_stopping=False,
                random_state=42,
            ),
        ),
        "release_catboost": (
            "release",
            CatBoostClassifier(
                iterations=400,
                depth=6,
                learning_rate=0.06,
                l2_leaf_reg=20,
                loss_function="Logloss",
                random_seed=42,
                thread_count=2,
                verbose=False,
                allow_writing_files=False,
            ),
        ),
        "release_rbf": (
            "release",
            make_pipeline(
                StandardScaler(),
                Nystroem(kernel="rbf", gamma=0.03, n_components=128, random_state=42),
                LogisticRegression(C=1, max_iter=500, tol=1e-3, random_state=42),
            ),
        ),
    }


def balanced_threshold(y, p):
    fpr, tpr, values = roc_curve(y, p)
    valid = np.flatnonzero(np.isfinite(values) & (values >= 0) & (values <= 1))
    best = min(valid, key=lambda i: (-(tpr[i] - fpr[i]), abs(values[i] - 0.5)))
    return float(values[best])


def precision_threshold(y, p, target=0.8, min_rows=500):
    order = np.argsort(-p, kind="stable")
    a = np.asarray(y)[order]
    v = p[order]
    cumulative = np.cumsum(a)
    n = np.arange(1, len(a) + 1)
    ties = np.r_[v[:-1] != v[1:], True]
    usable = np.flatnonzero(ties & (n >= min_rows) & (cumulative / n >= target))
    if not len(usable):
        return None
    return float(v[usable[-1]])


def selective_metrics(y, p, threshold):
    if threshold is None:
        return {
            "status": "not_available",
            "coverage": 0.0,
            "predicted_positive": 0,
            "precision": None,
        }
    selected = p >= threshold
    n = int(selected.sum())
    return {
        "status": "evaluated",
        "threshold": threshold,
        "coverage": float(selected.mean()),
        "predicted_positive": n,
        "precision": float(np.mean(y[selected])) if n else None,
        "recall": float(np.sum(y[selected]) / np.sum(y)) if np.sum(y) else None,
    }


def severity_model():
    return HistGradientBoostingRegressor(
        loss="poisson",
        max_iter=160,
        max_leaf_nodes=15,
        min_samples_leaf=100,
        l2_regularization=10,
        early_stopping=False,
        random_state=42,
    )


def regression_metrics(y, p, baseline):
    return {
        "rows": len(y),
        "mae": float(mean_absolute_error(y, p)),
        "rmse": float(mean_squared_error(y, p) ** 0.5),
        "baseline_mae": float(mean_absolute_error(y, np.full(len(y), baseline))),
        "baseline_rmse": float(mean_squared_error(y, np.full(len(y), baseline)) ** 0.5),
    }


def train(
    events=ROOT / "data" / "events.parquet",
    context=ROOT / "data" / "release_context.parquet",
    output=DEFAULT_MODEL,
    folds=5,
    include_kernel=True,
):
    events, context, output = map(Path, (events, context, output))
    table = flash_table(events, context)
    x = matrix(table)
    y = table["impact"].to_numpy().astype(np.int8)
    groups = table["match_id"].to_numpy()
    players = table["player_id"].to_numpy()
    count = table["enemy_blinded_players"].to_numpy().astype(float)
    amount = table["enemy_flash_added"].to_numpy().astype(float)
    if not np.array_equal(y, (count > 0).astype(np.int8)) or not np.isfinite(count).all():
        raise ValueError("Blind count and binary label disagree")
    tr, te, cv, kind = split_indices(x, y, groups, 42, folds)
    prototypes = candidates()
    if not include_kernel:
        prototypes.pop("release_rbf", None)
    all_oof = {}
    scores = {}
    with threadpool_limits(limits=2):
        for name, (profile, prototype) in prototypes.items():
            xx = x[:, : len(SETUP_FEATURES)] if profile == "setup" else x
            oof = np.full(len(tr), np.nan)
            fold_scores = []
            for i, (a, b) in enumerate(cv):
                fitted = clone(prototype).fit(xx[tr[a]], y[tr[a]])
                oof[b] = fitted.predict_proba(xx[tr[b]])[:, 1]
                fold_scores.append(metrics(y[tr[b]], oof[b]))
                print(
                    f"{name}: fold {i + 1}/{folds}; logloss {fold_scores[-1]['log_loss']:.6f}",
                    flush=True,
                )
            scores[name] = {
                "profile": profile,
                "oof": metrics(y[tr], oof),
                "folds": fold_scores,
            }
            all_oof[name] = oof
        selected = min(scores, key=lambda name: (scores[name]["oof"]["log_loss"], name))
        profile, prototype = prototypes[selected]
        xx = x[:, : len(SETUP_FEATURES)] if profile == "setup" else x
        threshold = balanced_threshold(y[tr], all_oof[selected])
        high = precision_threshold(y[tr], all_oof[selected])
        eval_model = clone(prototype).fit(xx[tr], y[tr])
        ptest = eval_model.predict_proba(xx[te])[:, 1]
        oof_amount = np.full(len(tr), np.nan)
        oof_count = np.full(len(tr), np.nan)
        for i, (a, b) in enumerate(cv):
            positive = tr[a][y[tr[a]] == 1]
            ah = severity_model().fit(x[positive], amount[positive])
            ch = severity_model().fit(x[positive], count[positive])
            oof_amount[b] = ah.predict(x[tr[b]])
            oof_count[b] = np.clip(ch.predict(x[tr[b]]), 1, 5)
            print(f"Severity heads: fold {i + 1}/{folds}", flush=True)
        positive = tr[y[tr] == 1]
        eval_amount = severity_model().fit(x[positive], amount[positive])
        eval_count = severity_model().fit(x[positive], count[positive])
        test_amount = eval_amount.predict(x[te])
        test_count = np.clip(eval_count.predict(x[te]), 1, 5)
        # Setup ablation uses exactly the same held-out matches.
        baseline_model = clone(prototypes["setup_catboost"][1]).fit(
            x[tr, : len(SETUP_FEATURES)], y[tr]
        )
        baseline_probability = baseline_model.predict_proba(x[te, : len(SETUP_FEATURES)])[:, 1]
        production = clone(prototype).fit(xx, y)
        positive_all = np.flatnonzero(y == 1)
        deploy_amount = severity_model().fit(x[positive_all], amount[positive_all])
        deploy_count = severity_model().fit(x[positive_all], count[positive_all])
    prior = float(y[tr].mean())
    baseline = np.full(len(te), prior)
    records = table.select(
        [
            "source_row_index",
            "match_id",
            "player_id",
            "round_index",
            "projectile_id",
            "side",
            "impact",
            "enemy_flash_added",
            "enemy_blinded_players",
        ]
    )
    partition = np.full(len(table), "train", dtype="<U5")
    partition[te] = "test"
    p = np.empty(len(table))
    p[tr] = all_oof[selected]
    p[te] = ptest
    severity = np.empty(len(table))
    severity[tr] = oof_amount
    severity[te] = test_amount
    counts = np.empty(len(table))
    counts[tr] = oof_count
    counts[te] = test_count
    fold = np.full(len(table), -1, dtype=np.int8)
    for i, (_, b) in enumerate(cv):
        fold[tr[b]] = i
    for name, values in [
        ("partition", partition),
        ("probability", p),
        ("conditional_flash_added", severity),
        ("conditional_blinded_players", counts),
        ("expected_flash_added", p * severity),
        ("expected_blinded_players", p * counts),
    ]:
        records = records.append_column(name, pa.array(values))
    records = records.append_column("validation_fold", pa.array(fold, mask=fold < 0))
    metrics_test = metrics(y[te], ptest, threshold)
    report = {
        "schema_version": VERSION,
        "selected_model": selected,
        "classification_profile": profile,
        "features": FEATURES if profile == "release" else SETUP_FEATURES,
        "severity_features": FEATURES,
        "feature_allowlist": SETUP_FEATURES + RELEASE_FEATURES,
        "feature_time": "first recorded release state; only strictly earlier releases in same round",
        "decision_threshold": threshold,
        "high_precision_threshold": high,
        "high_precision_target": 0.8,
        "threshold_selection": "Training OOF only: balanced accuracy; positive-only selective threshold targets empirical precision >=0.8 with >=500 examples. No guaranteed precision.",
        "source": {
            "events_sha256": sha256_file(events),
            "context_sha256": sha256_file(context),
        },
        "sample_sha256": fingerprint(x, y, groups),
        "selection": {
            "criterion": "minimum training OOF log loss",
            "candidates": scores,
            "fold_count": folds,
            "fold_validation_match_ids": [sorted(set(groups[tr[b]])) for _, b in cv],
        },
        "train_match_ids": sorted(set(groups[tr])),
        "test_match_ids": sorted(set(groups[te])),
        "split": kind,
        "train": {
            "rows": len(tr),
            "matches": len(set(groups[tr])),
            "prevalence": prior,
            "oof_metrics": metrics(y[tr], all_oof[selected], threshold),
        },
        "test": {
            "rows": len(te),
            "matches": len(set(groups[te])),
            "prevalence": float(y[te].mean()),
            "metrics": metrics_test,
            "metrics_at_0_5": metrics(y[te], ptest),
            "selective_positive": selective_metrics(y[te], ptest, high),
        },
        "baseline_constant": metrics(y[te], baseline),
        "baseline_setup": metrics(y[te], baseline_probability, threshold),
        "severity": {
            "conditional_flash_added": regression_metrics(
                amount[te][y[te] == 1],
                test_amount[y[te] == 1],
                float(amount[positive].mean()),
            ),
            "expected_flash_added": regression_metrics(
                amount[te], ptest * test_amount, float(amount[tr].mean())
            ),
            "expected_blinded_players": regression_metrics(
                count[te], ptest * test_count, float(count[tr].mean())
            ),
            "flash_added_unit": "unconfirmed native source unit",
            "blinded_players_unit": "distinct enemies with positive added flash; retrospective registered count",
        },
        "unseen_players": {
            "test_throwers": len(set(players[te])),
            "test_rows_from_unseen_throwers": 0,
        },
        "deployment": {
            "rows": len(table),
            "matches": len(set(groups)),
            "status": "refitted on all known outcomes after selection; held-out scores use the separate evaluation model",
        },
        "environment": {
            "numpy": np.__version__,
            "scikit_learn": __import__("sklearn").__version__,
            "catboost": __import__("catboost").__version__,
        },
        "limitations": [
            "Internal holdout was already inspected in earlier iterations; no new external evaluation",
            "Unknown outcomes are not imputed; predictions concern the registered-outcome population",
            "No enemy positions, round winners or causal grenade contribution in source",
            "Past releases are observer information and may be unavailable to a live player",
            "Initial projectile velocity and throw strength are not established by this source",
        ],
    }
    # Avoid repeatedly constructing a large set in the diagnostic above.
    train_players = set(players[tr])
    unseen = np.array([pid not in train_players for pid in players[te]])
    report["unseen_players"]["test_rows_from_unseen_throwers"] = int(unseen.sum())
    report["unseen_players"]["metrics"] = (
        metrics(y[te][unseen], ptest[unseen], threshold) if unseen.any() else None
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "evaluation": {
            "classification": eval_model,
            "amount": eval_amount,
            "count": eval_count,
        },
        "deployment": {
            "classification": production,
            "amount": deploy_amount,
            "count": deploy_count,
        },
        "classification_profile": profile,
    }
    joblib.dump(bundle, output, compress=3)
    report["artifact"] = {"filename": output.name, "sha256": sha256_file(output)}
    for name, values in all_oof.items():
        column = np.full(len(table), np.nan)
        column[tr] = values
        records = records.append_column(
            "candidate_" + name, pa.array(column, mask=~np.isfinite(column))
        )
    setup = np.empty(len(table))
    setup[tr] = all_oof["setup_catboost"]
    setup[te] = baseline_probability
    records = records.append_column("setup_probability", pa.array(setup))
    pq.write_table(records, output.with_suffix(".validation.parquet"), compression="zstd")
    report["predictions_sha256"] = sha256_file(output.with_suffix(".validation.parquet"))
    output.with_suffix(".metadata.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    complete_bundle(output, table=table, x=x)
    print(
        json.dumps(
            {
                "selected": selected,
                "test": report["test"],
                "severity": report["severity"],
            },
            indent=2,
        ),
        flush=True,
    )
    return json.loads(output.with_suffix(".metadata.json").read_text())


def complete_bundle(path=DEFAULT_MODEL, table=None, x=None):
    path = Path(path)
    report = json.loads(path.with_suffix(".metadata.json").read_text())
    bundle = joblib.load(path)
    if table is None:
        table = flash_table()
    if x is None:
        x = matrix(table)
    y = table["impact"].to_numpy().astype(np.int8)
    positive = y == 1
    actual_count = table["enemy_blinded_players"].to_numpy()
    if actual_count.max() > 5 or actual_count.min() < 0:
        raise ValueError("Source exceeds the declared five-opponent count support")
    if "setup_deployment" not in bundle:
        trained = np.isin(table["match_id"].to_numpy(), report["train_match_ids"])
        setup_prototype = candidates()["setup_catboost"][1]
        with threadpool_limits(limits=2):
            classifier = clone(setup_prototype).fit(x[trained, : len(SETUP_FEATURES)], y[trained])
            amount = severity_model().fit(
                x[trained & positive, : len(SETUP_FEATURES)],
                table["enemy_flash_added"].to_numpy()[trained & positive],
            )
            bundle["setup_deployment"] = {
                "classification": clone(classifier).fit(x[:, : len(SETUP_FEATURES)], y),
                "amount": clone(amount).fit(
                    x[positive, : len(SETUP_FEATURES)],
                    table["enemy_flash_added"].to_numpy()[positive],
                ),
            }
        bundle["setup_evaluation"] = {"classification": classifier, "amount": amount}
        report["setup_evaluation"] = {
            "metrics": report["baseline_setup"],
            "training_rows": int(trained.sum()),
            "deployment_rows": len(table),
            "interpretation": "Frozen setup model and severity parameters; deployment refitted on all known outcomes",
        }
    candidate_oof = pq.read_table(
        path.with_suffix(".validation.parquet"), filters=[("partition", "=", "train")]
    )
    report["setup_decision_threshold"] = balanced_threshold(
        candidate_oof["impact"].to_numpy(), candidate_oof["candidate_setup_catboost"].to_numpy()
    )
    heldout = np.isin(table["match_id"].to_numpy(), report["test_match_ids"])
    with threadpool_limits(limits=2):
        setup_test = bundle["setup_evaluation"]["classification"].predict_proba(
            x[heldout, : len(SETUP_FEATURES)]
        )[:, 1]
    report["setup_evaluation"]["metrics"] = metrics(
        y[heldout], setup_test, report["setup_decision_threshold"]
    )
    predictions_path = path.with_suffix(".validation.parquet")
    records = pq.read_table(predictions_path)
    count = np.clip(records["conditional_blinded_players"].to_numpy(), 1, 5)
    probability = records["probability"].to_numpy()
    for name, values in [
        ("conditional_blinded_players", count),
        ("expected_blinded_players", count * probability),
    ]:
        records = records.set_column(records.schema.get_field_index(name), name, pa.array(values))
    test = records["partition"].to_numpy() == "test"
    train = ~test
    report["severity"]["expected_blinded_players"] = regression_metrics(
        actual_count[test],
        (count * probability)[test],
        float(actual_count[train].mean()),
    )
    report["severity"]["conditional_count_postprocessing"] = (
        "Clip positive conditional mean to [1,5], the five-opponent support; unconditional expectation remains probability times conditional mean"
    )
    pq.write_table(records, predictions_path, compression="zstd")
    joblib.dump(bundle, path, compress=3)
    report["artifact"]["sha256"] = sha256_file(path)
    report["predictions_sha256"] = sha256_file(predictions_path)
    path.with_suffix(".metadata.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    return report


def load(path=DEFAULT_MODEL):
    path = Path(path)
    meta = json.loads(path.with_suffix(".metadata.json").read_text())
    if meta.get("schema_version") != VERSION or sha256_file(path) != meta["artifact"]["sha256"]:
        raise ValueError("Release model schema/checksum mismatch")
    if (
        __import__("sklearn").__version__ != meta["environment"]["scikit_learn"]
        or __import__("catboost").__version__ != meta["environment"]["catboost"]
    ):
        raise ValueError("Use versions pinned in requirements.txt")
    return joblib.load(path), meta


def probabilities(bundle, x, partition="deployment"):
    models = bundle[partition]
    xx = x[:, : len(SETUP_FEATURES)] if bundle["classification_profile"] == "setup" else x
    with threadpool_limits(limits=2):
        p = models["classification"].predict_proba(xx)[:, 1]
        amount = models["amount"].predict(x)
        count = np.clip(models["count"].predict(x), 1, 5)
    return p, amount, count


def predict(rows, bundle=None, metadata=None):
    if bundle is None:
        bundle, metadata = load()
    x = inputs(rows)
    p, amount, count = probabilities(bundle, x)
    return [
        {
            "row": i + 1,
            "mode": "release",
            "registered_enemy_effect_probability": float(prob),
            "predicted_effect": int(prob >= metadata["decision_threshold"]),
            "expected_blinded_enemy_players": float(prob * c),
            "conditional_blinded_enemy_players": float(c),
            "conditional_positive_flash_added": float(a),
            "expected_registered_flash_added": float(prob * a),
            "decision_threshold": metadata["decision_threshold"],
            "high_precision_positive": bool(
                metadata["high_precision_threshold"] is not None
                and prob >= metadata["high_precision_threshold"]
            ),
            "high_precision_note": "Empirical training-CV operating point; held-out precision and coverage are in model metadata; no guarantee",
            "duration_unit": "unconfirmed native source unit",
            "tactical_success": None,
            "model_training_scope": "all known source outcomes; unknown outcomes remain unknown",
        }
        for i, (prob, a, c) in enumerate(zip(p, amount, count))
    ]


def predict_setup(rows, bundle=None, metadata=None):
    require_cache(rows)
    if bundle is None:
        bundle, metadata = load()
    x = setup_input(rows)
    models = bundle["setup_deployment"]
    with threadpool_limits(limits=2):
        p = models["classification"].predict_proba(x)[:, 1]
        amount = models["amount"].predict(x)
    threshold = metadata["setup_decision_threshold"]
    return [
        {
            "row": i + 1,
            "mode": "setup",
            "registered_enemy_effect_probability": float(v),
            "predicted_effect": int(v >= threshold),
            "decision_threshold": threshold,
            "expected_registered_flash_added": float(v * a),
            "conditional_positive_flash_added": float(a),
            "expected_blinded_enemy_players": None,
            "duration_unit": "unconfirmed native source unit",
            "tactical_success": None,
            "missing_context": "Release position and/or earlier grenade activity are unavailable; the setup feature profile is used",
        }
        for i, (v, a) in enumerate(zip(p, amount))
    ]
