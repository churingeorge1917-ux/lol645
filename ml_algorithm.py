"""Геометрия раскидок, проверка связей и обучение.
"""

from __future__ import annotations
import argparse
import csv
import hashlib
import json
import logging
import math
import pickle
import sqlite3
import tempfile
from array import array
from collections import Counter, defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
from enum import IntEnum
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent
DEFAULT_DATABASE = ROOT / "dataset.sqlite"
LOG = logging.getLogger(__name__)
SEED = 42
TOLERANCES = (10, 15)

PROFILE_SCHEMA = """
CREATE TABLE IF NOT EXISTS player_matches (
    match_id TEXT NOT NULL, player_id TEXT NOT NULL,
    total_grenades INTEGER NOT NULL, smokes INTEGER NOT NULL,
    flashbangs INTEGER NOT NULL, fires INTEGER NOT NULL, hes INTEGER NOT NULL,
    eligible_grenades INTEGER NOT NULL, known_outcomes INTEGER NOT NULL,
    positive_outcomes INTEGER NOT NULL, zero_effect_outcomes INTEGER NOT NULL,
    unknown_outcomes INTEGER NOT NULL,
    observed_impact_rate REAL, effect_coverage REAL,
    impact_lower_bound REAL, impact_upper_bound REAL,
    enemy_damage REAL NOT NULL, ally_damage REAL NOT NULL,
    self_damage REAL NOT NULL, enemy_flash_added REAL NOT NULL,
    ally_flash_added REAL NOT NULL, self_flash_added REAL NOT NULL,
    PRIMARY KEY (match_id, player_id)
)
"""
EFFECT_FIELDS = (
    "enemy_damage",
    "ally_damage",
    "self_damage",
    "enemy_flash_added",
    "ally_flash_added",
    "self_flash_added",
)


def match_profiles(match: dict) -> list[dict]:
    """Все гранаты каждого игрока за матч по текущему определению импакта.

    Дым входит в число бросков, но не в знаменатель доли эффекта.
    Игроки из состава без гранат также сохраняются. Это не общий КПД.
    """
    players = {
        str(player["StaticInfo"]["PlayerID"]): Counter() for player in match["Players"]
    }
    seen = set()
    for kind, column in (
        ("Smokes", "smokes"),
        ("Flashbangs", "flashbangs"),
        ("Fires", "fires"),
        ("HEs", "hes"),
    ):
        for grenade in match[kind]:
            info = grenade["GeneralInfo"]
            identity = info["RoundIndexMatch"], str(info["ProjectileID"])
            if identity in seen:
                raise ValueError(
                    f"Повтор гранаты в матче {match['MatchID']}: {identity}"
                )
            seen.add(identity)
            if info["Team"] not in (Team.T, Team.CT):
                raise ValueError(f"Неигровая сторона: {info['Team']}")
            count = players.setdefault(str(info["ThrowerID"]), Counter())
            count["total_grenades"] += 1
            count[column] += 1
            effect, _ = enemy_effect(grenade, kind)
            for field in EFFECT_FIELDS:
                count[field] += effect[field]
            if kind != "Smokes":
                count["eligible_grenades"] += 1
                count["known_outcomes"] += effect["impact"] is not None
                count["positive_outcomes"] += effect["impact"] == 1
                count["zero_effect_outcomes"] += effect["impact"] == 0
                count["unknown_outcomes"] += effect["impact"] is None
    rows = []
    for player_id, count in sorted(players.items()):
        total, known = count["eligible_grenades"], count["known_outcomes"]
        positive, unknown = count["positive_outcomes"], count["unknown_outcomes"]
        row = {"match_id": str(match["MatchID"]), "player_id": player_id}
        row.update(
            {
                name: int(count[name])
                for name in (
                    "total_grenades",
                    "smokes",
                    "flashbangs",
                    "fires",
                    "hes",
                    "eligible_grenades",
                    "known_outcomes",
                    "positive_outcomes",
                    "zero_effect_outcomes",
                    "unknown_outcomes",
                )
            }
        )
        row.update(
            observed_impact_rate=positive / known if known else None,
            effect_coverage=known / total if total else None,
            impact_lower_bound=positive / total if total else None,
            impact_upper_bound=(positive + unknown) / total if total else None,
        )
        row.update({name: float(count[name]) for name in EFFECT_FIELDS})
        rows.append(row)
    return rows


def insert_profiles(connection: sqlite3.Connection, rows: list[dict]) -> None:
    if rows:
        fields = list(rows[0])
        connection.executemany(
            f"INSERT INTO player_matches ({','.join(fields)}) "
            f"VALUES ({','.join('?' for _ in fields)})",
            [[row[field] for field in fields] for row in rows],
        )


def profile_summary(connection: sqlite3.Connection, source_hash: str) -> dict:
    totals = connection.execute(
        "SELECT COUNT(*), COUNT(DISTINCT match_id), SUM(total_grenades), "
        "SUM(eligible_grenades), SUM(known_outcomes), SUM(positive_outcomes), "
        "SUM(unknown_outcomes) FROM player_matches"
    ).fetchone()
    return dict(
        zip(
            (
                "player_matches",
                "matches",
                "throws",
                "eligible",
                "known",
                "positive",
                "unknown",
            ),
            totals,
        ),
        source_sha256=source_hash,
        definition="Любой положительный урон HP или добавленное ослепление противника; дым исключён из долей; пропуски неизвестны.",
        scope="Все гранаты, без фильтра близости к эталонам; обе стороны за матч суммарно.",
    )


def refresh_player_matches(source: Path, target: Path) -> dict:
    """Пересчитать полный профиль из исходника, не меняя модель и её выборку."""
    with database(target) as connection:
        read_document(connection, "source")
        extraction = read_document(connection, "extraction")
    with source.open("rb") as stream:
        source_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    with tempfile.TemporaryDirectory(
        prefix=".profiles-", dir=target.parent
    ) as directory:
        temporary = Path(directory) / "profiles.sqlite"
        prepared = sqlite3.connect(temporary)
        prepared.execute(PROFILE_SCHEMA)
        matches = set()

        def collect(match):
            match_id = str(match["MatchID"])
            if match_id in matches:
                raise ValueError(f"Повтор MatchID: {match_id}")
            matches.add(match_id)
            insert_profiles(prepared, match_profiles(match))
            if len(matches) % 500 == 0:
                LOG.info("Полные профили: обработано %s матчей", len(matches))

        try:
            read_matches(source, collect)
            summary = profile_summary(prepared, source_hash)
            if (
                summary["matches"] != extraction["matches"]
                or summary["throws"] != extraction["counts"]["throws"]
            ):
                raise ValueError(
                    "Число матчей или бросков расходится с исходной базой."
                )
            prepared.commit()
        finally:
            prepared.close()
        with database(target, writable=True) as connection:
            connection.execute("ATTACH DATABASE ? AS prepared", (str(temporary),))
            connection.execute(PROFILE_SCHEMA)
            connection.execute("DELETE FROM player_matches")
            connection.execute(
                "INSERT INTO player_matches SELECT * FROM prepared.player_matches"
            )
            save_document(connection, "player_profiles", summary)
    return summary


class Team(IntEnum):
    """Коды стороны в момент события, а не постоянные составы команд."""

    UNASSIGNED = 0
    SPECTATOR = 1
    T = 2
    CT = 3


@contextmanager
def database(path: Path, *, writable: bool = False) -> Iterator[sqlite3.Connection]:
    if not path.is_file():
        raise FileNotFoundError(path)
    connection = sqlite3.connect(
        path.resolve().as_uri() + ("?mode=rw" if writable else "?mode=ro"), uri=True
    )
    connection.row_factory = sqlite3.Row
    try:
        yield connection
        if writable:
            connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def read_document(connection: sqlite3.Connection, key: str) -> Any:
    row = connection.execute(
        "SELECT value FROM documents WHERE key = ?", (key,)
    ).fetchone()
    if row is None:
        raise KeyError(key)
    return json.loads(row[0])


def save_document(connection: sqlite3.Connection, key: str, value: Any) -> None:
    connection.execute(
        "INSERT OR REPLACE INTO documents(key, value) VALUES (?, ?)",
        (key, json.dumps(value, ensure_ascii=False, allow_nan=False)),
    )


def launch_matches(row: dict, tolerance: float) -> bool:
    """Совпадение позиции и направления для неподвижного броска стоя."""
    return bool(
        row["standing"]
        and row["position_error_pct"] <= tolerance
        and row["direction_error_pct"] <= tolerance
    )


def full_matches(row: dict, tolerance: float) -> bool | None:
    """Для эталона без конечной точки результат неизвестен."""
    if row["landing_error_pct"] is None:
        return None
    return launch_matches(row, tolerance) and row["landing_error_pct"] <= tolerance


def summarize_geometry(rows: list[dict]) -> list[dict]:
    groups = defaultdict(Counter)
    for row in rows:
        for tolerance in TOLERANCES:
            count = groups[(row["reference"], row["side"], tolerance)]
            count["nearby"] += 1
            count["standing"] += bool(row["standing"])
            count["launch"] += launch_matches(row, tolerance)
            full = full_matches(row, tolerance)
            count["endpoint_available"] += full is not None
            count["full"] += full is True
    result = [
        dict(reference=key[0], side=key[1], tolerance_pct=key[2], **count)
        for key, count in sorted(groups.items())
    ]
    for group in result:
        if not group["endpoint_available"]:
            group["full"] = None
    return result


def reference_associations(rows: list[dict], repetitions: int = 1000) -> list[dict]:
    """Phi и разность долей; интервал получаем пересэмплированием целых матчей."""
    groups = {}
    for row in rows:
        if not row["standing"]:
            continue
        for tolerance in TOLERANCES:
            key = row["reference"], row["side"], tolerance
            totals, matches = groups.setdefault(key, (Counter(), {}))
            within = launch_matches(row, tolerance)
            prefix = "within" if within else "outside"
            totals[prefix + "_total"] += 1
            if row["impact"] is None:
                continue
            positive = row["impact"]
            totals[prefix + "_known"] += 1
            totals[prefix + "_positive"] += positive
            table = matches.setdefault(row["match_id"], [0, 0, 0, 0])
            table[(0 if within else 2) + (0 if positive else 1)] += 1
    rng = np.random.default_rng(741)
    results = []
    for (reference, side, tolerance), (totals, matches) in sorted(groups.items()):
        table = np.asarray(list(matches.values()), dtype=np.int64).reshape(-1, 4)
        a, b, c, d = map(int, table.sum(axis=0)) if len(table) else (0, 0, 0, 0)
        denominator = math.sqrt(float((a + b) * (c + d) * (a + c) * (b + d)))
        within_matches = int(np.count_nonzero(table[:, :2].sum(axis=1)))
        outside_matches = int(np.count_nonzero(table[:, 2:].sum(axis=1)))
        draws = []
        if min(within_matches, outside_matches) >= 5 and a + c and b + d:
            for _ in range(repetitions):
                aa, bb, cc, dd = table[rng.integers(0, len(table), len(table))].sum(
                    axis=0
                )
                if aa + bb and cc + dd:
                    draws.append(float(aa / (aa + bb) - cc / (cc + dd)))
        low, high = (
            map(float, np.quantile(draws, [0.025, 0.975])) if draws else (None, None)
        )
        results.append(
            dict(
                reference=reference,
                side=side,
                tolerance_pct=tolerance,
                **{
                    f"{prefix}_{name}": int(totals[f"{prefix}_{name}"])
                    for prefix in ("within", "outside")
                    for name in ("total", "known", "positive")
                },
                within_matches=within_matches,
                outside_matches=outside_matches,
                phi=(a * d - b * c) / denominator if denominator else None,
                delta=a / (a + b) - c / (c + d) if a + b and c + d else None,
                delta_low=low,
                delta_high=high,
                bootstrap_repetitions=len(draws),
            )
        )
    return results


def probability_metrics(y: np.ndarray, probabilities: np.ndarray) -> dict:
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        balanced_accuracy_score,
        brier_score_loss,
        confusion_matrix,
        log_loss,
        roc_auc_score,
    )

    predicted = probabilities >= 0.5
    return {
        "roc_auc": float(roc_auc_score(y, probabilities)),
        "average_precision": float(average_precision_score(y, probabilities)),
        "accuracy": float(accuracy_score(y, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(y, predicted)),
        "brier": float(brier_score_loss(y, probabilities)),
        "log_loss": float(log_loss(y, probabilities, labels=[0, 1])),
        "confusion_matrix": confusion_matrix(y, predicted, labels=[0, 1]).tolist(),
    }


def fit_classifier(
    features: list[dict], labels: list[int], match_ids: list[str], *, seed: int = SEED
) -> tuple[dict, list[dict]]:
    from sklearn.feature_extraction import DictVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupShuffleSplit
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    groups, y = np.asarray(match_ids), np.asarray(labels, dtype=int)
    if len(set(match_ids)) < 10 or len(set(labels)) != 2:
        return {
            "status": "not_estimable",
            "reason": "Нужны два класса и не менее 10 матчей.",
        }, []
    split = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    train, test = next(split.split(features, y, groups))
    train_matches, test_matches = sorted(set(groups[train])), sorted(set(groups[test]))
    if set(train_matches) & set(test_matches):
        raise AssertionError("Пересечение матчей train и test")
    if min(len(set(y[train])), len(set(y[test]))) < 2:
        return {
            "status": "not_estimable",
            "reason": "В одной части разбиения нет двух классов.",
        }, []
    model = make_pipeline(
        DictVectorizer(sparse=False),
        StandardScaler(),
        LogisticRegression(C=1.0, max_iter=2000, random_state=seed),
    )
    model.fit([features[i] for i in train], y[train])
    train_probability = model.predict_proba([features[i] for i in train])[:, 1]
    test_probability = model.predict_proba([features[i] for i in test])[:, 1]
    prior = float(y[train].mean())
    result = {
        "status": "trained",
        "seed": seed,
        "split": "GroupShuffleSplit by match_id; test_size=0.2",
        "features": model[0].get_feature_names_out().tolist(),
        "train": {
            "rows": len(train),
            "matches": len(train_matches),
            "prevalence": prior,
            "metrics": probability_metrics(y[train], train_probability),
        },
        "test": {
            "rows": len(test),
            "matches": len(test_matches),
            "prevalence": float(y[test].mean()),
            "metrics": probability_metrics(y[test], test_probability),
        },
        "baseline": probability_metrics(y[test], np.full(len(test), prior)),
        "train_match_ids": train_matches,
        "test_match_ids": test_matches,
        "coefficients": model[2].coef_[0].tolist(),
        "intercept": float(model[2].intercept_[0]),
        "scaler_mean": model[1].mean_.tolist(),
        "scaler_scale": model[1].scale_.tolist(),
    }
    predictions = []
    for part, indices, probabilities in (
        ("train", train, train_probability),
        ("test", test, test_probability),
    ):
        predictions.extend(
            {
                "row_index": int(i),
                "match_id": str(groups[i]),
                "partition": part,
                "label": int(y[i]),
                "probability": float(p),
            }
            for i, p in zip(indices, probabilities)
        )
    return result, predictions


def flash_model(rows: list[dict]) -> tuple[dict, list[dict]]:
    eligible = [
        r for r in rows if r["type"] == "Flashbangs" and r["impact"] is not None
    ]
    numeric = (
        "position_error_pct",
        "direction_error_pct",
        "speed",
        "standing",
        "player_x",
        "player_y",
        "player_z",
    )
    features = [
        dict(
            **{name: float(r[name]) for name in numeric},
            reference=r["reference"],
            side=str(r["side"]),
        )
        for r in eligible
    ]
    result, predictions = fit_classifier(
        features, [r["impact"] for r in eligible], [r["match_id"] for r in eligible]
    )
    result["scope"] = (
        "Известные исходы Flashbangs рядом с тремя внешними эталонами; включает движение. Не все флешки датасета."
    )
    for prediction in predictions:
        row = eligible[prediction["row_index"]]
        prediction.update(projectile_id=row["projectile_id"], round_index=row["round"])
    return result, predictions


CONTEXT_COLUMNS = (
    "match_id",
    "round_index",
    "player_id",
    "side",
    "round_won",
    "player_kpd",
)


def load_context(path: Path, definition: str, source_matches: set[str]) -> list[dict]:
    """Проверить внешний независимый КПД и исходы до соединения таблиц."""
    if not definition.strip():
        raise ValueError(
            "Нужно описать независимое определение КПД через --kpd-definition."
        )
    rows, identities, outcomes = [], set(), {}
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not set(CONTEXT_COLUMNS) <= set(reader.fieldnames or []):
            raise ValueError("CSV должен содержать: " + ", ".join(CONTEXT_COLUMNS))
        for number, raw in enumerate(reader, start=2):
            if any(raw[name] is None for name in CONTEXT_COLUMNS):
                raise ValueError(f"Неполная строка CSV: {number}")
            row = {name: raw[name].strip() for name in CONTEXT_COLUMNS}
            for name in ("round_index", "side", "round_won"):
                row[name] = int(row[name])
            row["player_kpd"] = float(row["player_kpd"])
            if (
                row["side"] not in (2, 3)
                or row["round_won"] not in (0, 1)
                or row["round_index"] < 0
                or not math.isfinite(row["player_kpd"])
            ):
                raise ValueError(f"Некорректные значения в строке {number}")
            if row["match_id"] not in source_matches:
                raise ValueError(
                    f"Матч строки {number} отсутствует в подготовленной выборке."
                )
            key = row["match_id"], row["round_index"], row["player_id"]
            if not row["player_id"] or key in identities:
                raise ValueError(f"Пустой игрок или повтор ключа в строке {number}")
            identities.add(key)
            team_key = row["match_id"], row["round_index"], row["side"]
            if team_key in outcomes and outcomes[team_key] != row["round_won"]:
                raise ValueError(
                    f"Противоречащие исходы одной стороны в строке {number}"
                )
            other = team_key[:2] + (5 - row["side"],)
            if other in outcomes and outcomes[other] == row["round_won"]:
                raise ValueError(
                    f"Обе стороны имеют одинаковый исход в строке {number}"
                )
            outcomes[team_key] = row["round_won"]
            rows.append(row)
    if not rows:
        raise ValueError("Контекст пуст")
    return rows


def correlation_interval(rows: list[dict], x: str, y: str, repetitions: int) -> dict:
    """Pearson/Spearman; неопределённость Pearson с кластеризацией по матчу."""
    from scipy.stats import rankdata

    values = np.asarray([[r[x], r[y]] for r in rows], dtype=float).reshape(-1, 2)

    def coefficient(array: np.ndarray) -> float | None:
        if len(array) < 3 or np.any(np.ptp(array, axis=0) == 0):
            return None
        return float(np.corrcoef(array.T)[0, 1])

    pearson = coefficient(values)
    spearman = (
        coefficient(np.column_stack([rankdata(values[:, i]) for i in (0, 1)]))
        if len(values)
        else None
    )
    clusters = defaultdict(list)
    for i, row in enumerate(rows):
        clusters[row["match_id"]].append(i)
    blocks = list(clusters.values())
    rng, draws = np.random.default_rng(741), []
    if pearson is not None and len(blocks) >= 5:
        for _ in range(repetitions):
            indices = [
                i
                for block in rng.integers(0, len(blocks), len(blocks))
                for i in blocks[block]
            ]
            estimate = coefficient(values[indices])
            if estimate is not None:
                draws.append(estimate)
    interval = list(map(float, np.quantile(draws, [0.025, 0.975]))) if draws else None
    return {
        "x": x,
        "y": y,
        "n": len(rows),
        "matches": len(blocks),
        "pearson": pearson,
        "spearman": spearman,
        "pearson_ci95": interval,
    }


def player_rounds(rows: list[dict]) -> dict[tuple, dict]:
    """Гранаты только покрытой пулом части, а не полная статистика игрока."""
    counts = defaultdict(Counter)
    for row in rows:
        key = row["match_id"], row["round"], str(row["player_id"]), row["side"]
        count = counts[key]
        count["pool_throws"] += 1
        count["known_throws"] += row["impact"] is not None
        count["positive_throws"] += row["impact"] == 1
    return {
        key: dict(
            **count,
            utility_rate=count["positive_throws"] / count["known_throws"]
            if count["known_throws"]
            else None,
            effect_coverage=count["known_throws"] / count["pool_throws"],
        )
        for key, count in counts.items()
    }


def outcome_analysis(
    rows: list[dict], context: list[dict], definition: str, repetitions: int
) -> tuple[dict, list[dict]]:
    aggregates = player_rounds(rows)
    joined = []
    teams = defaultdict(list)
    for row in context:
        teams[(row["match_id"], row["round_index"], row["side"])].append(row)
        key = row["match_id"], row["round_index"], row["player_id"], row["side"]
        if key in aggregates and aggregates[key]["utility_rate"] is not None:
            joined.append(dict(**row, **aggregates[key]))
    pairs = (
        ("player_kpd", "utility_rate"),
        ("utility_rate", "round_won"),
        ("player_kpd", "round_won"),
    )
    correlations = [correlation_interval(joined, x, y, repetitions) for x, y in pairs]
    samples, features = [], []
    for (match_id, round_index, side), roster in sorted(teams.items()):
        # Для среднего КПД команды необходим полный состав, по одному исходу на сторону-раунд.
        if len(roster) != 5:
            continue
        utility = [
            aggregates.get((match_id, round_index, r["player_id"], side), {})
            for r in roster
        ]
        known = sum(r.get("known_throws", 0) for r in utility)
        total = sum(r.get("pool_throws", 0) for r in utility)
        if not known:
            continue
        features.append(
            {
                "player_kpd_mean": float(np.mean([r["player_kpd"] for r in roster])),
                "utility_rate": sum(r.get("positive_throws", 0) for r in utility)
                / known,
                "effect_coverage": known / total,
                "pool_throws": total,
                "side": str(side),
            }
        )
        samples.append(
            {
                "match_id": match_id,
                "round_index": round_index,
                "side": side,
                "label": roster[0]["round_won"],
            }
        )
    model, predictions = fit_classifier(
        features, [r["label"] for r in samples], [r["match_id"] for r in samples]
    )
    for prediction in predictions:
        prediction.update(samples[prediction["row_index"]])
    return {
        "status": "context_loaded",
        "kpd_definition": definition,
        "context_rows": len(context),
        "joined_player_rounds": len(joined),
        "correlations": correlations,
        "round_model": model,
        "interpretation": "Ретроспективная связь по итогам раунда; не причинный эффект и не прогноз до начала раунда.",
    }, predictions


def analyze(
    path: Path, context_path: Path | None, definition: str, repetitions: int
) -> dict:
    with database(path, writable=True) as connection:
        rows = [
            dict(r)
            for r in connection.execute(
                "SELECT * FROM throws ORDER BY match_id, round, projectile_id"
            )
        ]
        LOG.info(
            "Данные: %s ближайших бросков; исходный охват %s матчей",
            len(rows),
            read_document(connection, "extraction")["matches"],
        )
        geometry = summarize_geometry(rows)
        associations = reference_associations(rows, repetitions)
        flash, predictions = flash_model(rows)
        if context_path is not None:
            context = load_context(
                context_path, definition, {r["match_id"] for r in rows}
            )
            outcomes, round_predictions = outcome_analysis(
                rows, context, definition, repetitions
            )
            save_document(connection, "context", context)
        else:
            outcomes = {
                "status": "missing_context",
                "reason": "В исходнике нет независимого КПД и победителей раундов. Нужен внешний CSV.",
            }
            round_predictions = []
            connection.execute("DELETE FROM documents WHERE key = 'context'")
        for key, value in (
            ("geometry", geometry),
            ("associations", associations),
            ("flash_model", flash),
            ("flash_predictions", predictions),
            ("outcomes", outcomes),
            ("round_predictions", round_predictions),
        ):
            save_document(connection, key, value)
    return {"flash": flash.get("test"), "round_outcomes": outcomes}


class ObjectId:
    def __setstate__(self, state):
        if not isinstance(state, bytes):
            raise pickle.UnpicklingError("Unexpected ObjectId state")
        self.value = state


class BoundedMemo:

    def __init__(self):
        self.index = array("i")
        self.atoms = []
        self.atom_ids = {}
        self.objects = {}

    def __len__(self):
        return len(self.index)

    def __setitem__(self, key, value):
        code = -1
        if isinstance(value, (str, bytes, type)):
            atom_key = (type(value), value)
            if atom_key not in self.atom_ids:
                self.atom_ids[atom_key] = len(self.atoms)
                self.atoms.append(value)
            code = self.atom_ids[atom_key]
        else:
            self.objects[key] = value
        if key == len(self.index):
            self.index.append(code)
        elif key < len(self.index):
            self.index[key] = code
        else:
            raise pickle.UnpicklingError("Non-contiguous memo index")

    def __getitem__(self, key):
        code = self.index[key]
        if code >= 0:
            return self.atoms[code]
        if key in self.objects:
            return self.objects[key]
        raise pickle.UnpicklingError(
            f"Cross-match reference to evicted memo entry {key}"
        )

    def prune(self, active):
        for key in [k for k, v in self.objects.items() if id(v) not in active]:
            del self.objects[key]


class MatchReader(pickle._Unpickler):
    def __init__(self, file, callback):
        super().__init__(file)
        self.memo = BoundedMemo()
        self.callback = callback

    def find_class(self, module, name):
        if (module, name) == ("bson.int64", "Int64"):
            return int
        if (module, name) == ("bson.objectid", "ObjectId"):
            return ObjectId
        raise pickle.UnpicklingError(f"Global is not allowed: {module}.{name}")

    def load_setitems(self):
        super().load_setitems()
        value = self.stack[-1]
        if isinstance(value, dict) and {"MatchID", "Players"} <= value.keys():
            self.callback(value)
            value.clear()
            active = {id(v) for v in self.stack}
            active.update(id(v) for stack in self.metastack for v in stack)
            self.memo.prune(active)


MatchReader.dispatch = pickle._Unpickler.dispatch.copy()
MatchReader.dispatch[pickle.SETITEMS[0]] = MatchReader.load_setitems


def read_matches(path, callback):
    """Call callback(match) for each match; do not retain the passed dictionary."""
    with open(path, "rb") as source:
        MatchReader(source, callback).load()


def vector(point):
    return tuple(float(point[axis]) for axis in "XYZ")


def direction(pitch, yaw):
    pitch, yaw = math.radians(pitch), math.radians(yaw)
    return (
        math.cos(pitch) * math.cos(yaw),
        math.cos(pitch) * math.sin(yaw),
        -math.sin(pitch),
    )


def direction_error(pitch, yaw, reference):
    chord = math.dist(
        direction(pitch, yaw), direction(reference["pitch"], reference["yaw"])
    )
    return 100 * chord, math.degrees(2 * math.asin(min(1, chord / 2)))


def cell(point):
    return tuple(math.floor(value / size) for value, size in zip(point, (256, 256, 64)))


def enemy_effect(grenade, kind):
    info = grenade["GeneralInfo"]
    result = {
        "impact": None,
        "enemy_damage": 0,
        "enemy_flash_added": 0.0,
        "ally_damage": 0,
        "self_damage": 0,
        "ally_flash_added": 0.0,
        "self_flash_added": 0.0,
    }
    positions = []
    field = "FlashedPlayers" if kind == "Flashbangs" else "DamagedPlayers"
    if kind == "Smokes" or field not in grenade:
        return result, positions
    valid = info["Team"] in (Team.T, Team.CT)
    durations = defaultdict(dict)
    for target in grenade[field]:
        valid &= target["Team"] in (Team.T, Team.CT)
        group = (
            "self"
            if target["PlayerID"] == info["ThrowerID"]
            else "ally"
            if target["Team"] == info["Team"]
            else "enemy"
        )
        if kind == "Flashbangs":
            value = target.get("FlashDurationAdded")
            if (
                not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
            ):
                valid = False
                continue
            durations[group][target["PlayerID"]] = max(
                value, durations[group].get(target["PlayerID"], 0)
            )
            if group == "enemy":
                positions.append(
                    {
                        "player_id": target["PlayerID"],
                        "position": vector(target["Position"]),
                        "value": value,
                        "unit": "FlashDurationAdded",
                    }
                )
        else:
            for hit in target["Damage"]:
                value = hit.get("HealthDamage")
                if (
                    not isinstance(value, (int, float))
                    or not math.isfinite(value)
                    or value < 0
                ):
                    valid = False
                    continue
                result[group + "_damage"] += value
                if group == "enemy":
                    positions.append(
                        {
                            "player_id": target["PlayerID"],
                            "position": vector(hit["Position"]),
                            "value": value,
                            "tick": hit["TickNumber"],
                            "unit": "HP",
                        }
                    )
    for group, values in durations.items():
        result[group + "_flash_added"] = sum(values.values())
    if valid:
        result["impact"] = int(
            result["enemy_damage"] > 0 or result["enemy_flash_added"] > 0
        )
    return result, positions


def nearest_reference(info, kind, references, radius=300):
    point = vector(info["ThrowPoint"]["Position"])
    candidates = [
        ref
        for ref in references
        if ref["type"] == kind
        and ref["side"] in (None, info["Team"])
        and math.dist(point[:2], ref["setup"][:2]) <= radius
        and abs(point[2] - ref["setup"][2]) <= 256
    ]
    # Назначение по месту броска, без использования воздействия или конечной точки.
    return min(
        candidates,
        key=lambda ref: (math.dist(point, ref["setup"]), ref["id"]),
        default=None,
    )


def compare(info, reference, scale=1000):
    point = info["ThrowPoint"]
    # Вывод о порядке углов проверяется отдельно по горизонтальному смещению PosFrom.
    error, degrees = direction_error(point["ViewY"], point["ViewX"], reference)
    speed = math.dist(vector(point["Velocity"]), (0, 0, 0))
    return {
        "position_error_pct": 100
        * math.dist(vector(point["Position"]), reference["setup"])
        / scale,
        "direction_error_pct": error,
        "angle_error_deg": degrees,
        "landing_error_pct": (
            100 * math.dist(vector(info["PosTo"]), reference["destination"]) / scale
            if reference["destination"] is not None
            else None
        ),
        "standing": int(
            not point["IsCrouching"] and not point["IsAirborne"] and speed <= 20
        ),
        "speed": speed,
    }


def rebuild_database(source: Path, target: Path) -> None:
    temporary = target.with_suffix(".building.sqlite")
    if temporary.exists():
        raise FileExistsError(f"Незавершённая сборка уже существует: {temporary}")
    with database(target) as original:
        catalog = read_document(original, "references")
        rebuilt = sqlite3.connect(temporary)
        original.backup(rebuilt)
    references = catalog["references"]
    counts, grid, examples = Counter(), defaultdict(Counter), {}
    matches = set()
    bounds = [[math.inf, -math.inf] for _ in range(3)]
    rebuilt.execute("DELETE FROM throws")
    rebuilt.execute("DELETE FROM spatial_grid")
    rebuilt.execute(PROFILE_SCHEMA)
    rebuilt.execute("DELETE FROM player_matches")
    rebuilt.execute("DELETE FROM documents WHERE key != 'references'")

    def collect(match):
        match_id = str(match["MatchID"])
        if match_id in matches:
            raise ValueError(f"Повтор MatchID: {match_id}")
        matches.add(match_id)
        insert_profiles(rebuilt, match_profiles(match))
        seen = set()
        for kind in ("Smokes", "Flashbangs", "Fires", "HEs"):
            for grenade in match[kind]:
                info = grenade["GeneralInfo"]
                identity = info["RoundIndexMatch"], str(info["ProjectileID"])
                if identity in seen:
                    raise ValueError(f"Повтор гранаты в матче {match_id}: {identity}")
                seen.add(identity)
                points = {
                    "player": vector(info["ThrowPoint"]["Position"]),
                    "start": vector(info["PosFrom"]),
                    "end": vector(info["PosTo"]),
                }
                if not all(
                    math.isfinite(v) for point in points.values() for v in point
                ):
                    raise ValueError(f"Неконечная координата в матче {match_id}")
                side = info["Team"]
                if side not in (Team.T, Team.CT):
                    raise ValueError(f"Неигровая сторона {side}")
                effect, targets = enemy_effect(grenade, kind)
                counts["throws"] += 1
                counts[f"{side}/{kind}/all"] += 1
                counts[
                    f"{side}/{kind}/"
                    + ("unknown" if effect["impact"] is None else "known")
                ] += 1
                counts[f"{side}/{kind}/positive"] += effect["impact"] == 1
                for name in ("player", "end"):
                    key = (side, kind, name) + cell(points[name])
                    grid[key]["throws"] += 1
                    grid[key]["known"] += effect["impact"] is not None
                    grid[key]["positive"] += effect["impact"] == 1
                for i, value in enumerate(points["player"]):
                    bounds[i][0] = min(bounds[i][0], value)
                    bounds[i][1] = max(bounds[i][1], value)
                reference = nearest_reference(info, kind, references)
                if reference is None:
                    continue
                row = dict(
                    match_id=match_id,
                    player_id=str(info["ThrowerID"]),
                    round=info["RoundIndexMatch"],
                    projectile_id=str(info["ProjectileID"]),
                    side=side,
                    type=kind,
                    reference=reference["id"],
                    **effect,
                    **compare(info, reference, catalog["position_scale_units"]),
                )
                for name, point in points.items():
                    row.update(
                        {f"{name}_{axis}": value for axis, value in zip("xyz", point)}
                    )
                fields = list(row)
                rebuilt.execute(
                    f"INSERT INTO throws ({','.join(fields)}) VALUES ({','.join('?' for _ in fields)})",
                    list(row.values()),
                )
                counts["nearby"] += 1
                key = f"{reference['id']}/{side}"
                if row["standing"] and (kind == "Smokes" or effect["impact"] == 1):
                    score = max(row["position_error_pct"], row["direction_error_pct"])
                    if key not in examples or score < examples[key]["selection_score"]:
                        examples[key] = {
                            "selection_score": score,
                            "throw": row,
                            "targets": targets,
                        }
        if len(matches) % 250 == 0:
            LOG.info("Прочитано %s матчей, %s бросков", len(matches), counts["throws"])

    try:
        read_matches(source, collect)
        rebuilt.executemany(
            "INSERT INTO spatial_grid VALUES (?,?,?,?,?,?,?,?,?)",
            [
                list(key) + [values[name] for name in ("throws", "known", "positive")]
                for key, values in sorted(grid.items())
            ],
        )
        with source.open("rb") as stream:
            source_hash = hashlib.file_digest(stream, "sha256").hexdigest()
        metadata = {
            "matches": len(matches),
            "source_bytes": source.stat().st_size,
            "counts": dict(counts),
            "grid_size": [256, 256, 64],
            "coordinate_bounds": bounds,
            "reference_count": len(references),
            "nearby_radius_xy": 300,
            "standing_max_speed": 20,
            "position_scale_units": catalog["position_scale_units"],
            "map_version_compatibility": "unverified",
            "complete_source": True,
        }
        for key, value in (
            (
                "source",
                {
                    "filename": source.name,
                    "bytes": source.stat().st_size,
                    "sha256": source_hash,
                    "observed_unique_matches": len(matches),
                },
            ),
            ("extraction", metadata),
            ("examples", examples),
            ("player_profiles", profile_summary(rebuilt, source_hash)),
        ):
            save_document(rebuilt, key, value)
        rebuilt.commit()
    except Exception:
        rebuilt.close()
        temporary.unlink(missing_ok=True)
        raise
    rebuilt.close()
    temporary.replace(target)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATABASE)
    preparation = parser.add_mutually_exclusive_group()
    preparation.add_argument(
        "--rebuild-source",
        type=Path,
        help="Повторить подготовку из исходного pickle; длительная операция",
    )
    preparation.add_argument(
        "--refresh-player-matches",
        type=Path,
        help="Пересчитать полную гранатную статистику игроков из исходного pickle",
    )
    parser.add_argument(
        "--context", type=Path, help="CSV с независимым КПД и исходами раундов"
    )
    parser.add_argument(
        "--kpd-definition",
        default="",
        help="Содержательное определение внешнего КПД и периода его расчёта",
    )
    parser.add_argument("--bootstrap", type=int, default=1000)
    args = parser.parse_args()
    if args.bootstrap < 1:
        parser.error("--bootstrap должен быть положительным")
    if args.context is not None and not args.kpd_definition.strip():
        parser.error("Для --context требуется --kpd-definition")
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if args.refresh_player_matches is not None:
        if args.context is not None:
            parser.error("Обновление профилей запускается отдельно от --context")
        print(
            json.dumps(
                refresh_player_matches(args.refresh_player_matches, args.data),
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    if args.rebuild_source is not None:
        rebuild_database(args.rebuild_source, args.data)
    print(
        json.dumps(
            analyze(args.data, args.context, args.kpd_definition, args.bootstrap),
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
