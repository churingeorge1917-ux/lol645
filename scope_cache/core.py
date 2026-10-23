"""Общие проверки, метрики и чтение исходных матчей."""

from __future__ import annotations
import hashlib
import json
import math
import pickle
from array import array
from collections import defaultdict
from enum import IntEnum
from pathlib import Path
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold, TimeSeriesSplit
from scope_cache import ROOT


class Team(IntEnum):
    """Коды стороны в момент события, а не постоянные составы команд."""

    UNASSIGNED = 0
    SPECTATOR = 1
    T = 2
    CT = 3


class ObjectId:
    def __setstate__(self, state):
        if not isinstance(state, bytes):
            raise pickle.UnpicklingError("Unexpected ObjectId state")
        self.value = state


class BoundedMemo:
    """Compact memo: repeated strings share atoms, finished containers expire.

    Keeping a Python dict entry for every repeated BSON key consumes much more
    RAM than the events themselves. A 32-bit array preserves original pickle
    memo indexes; a small atom table shares strings/bytes/classes by value.
    """

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
        raise pickle.UnpicklingError(f"Cross-match reference to evicted memo entry {key}")

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


def enemy_effect(grenade, kind):
    """Пропущенный список целей остаётся неизвестным; союзники и сам игрок — отдельно."""
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
    if kind == "Smokes" or not isinstance(grenade.get(field), list):
        return result, positions
    valid = info["Team"] in (Team.T, Team.CT)
    durations = defaultdict(dict)
    for target in grenade[field]:
        valid &= target["Team"] in (Team.T, Team.CT)
        group = (
            "self"
            if str(target["PlayerID"]) == str(info["ThrowerID"])
            else "ally"
            if target["Team"] == info["Team"]
            else "enemy"
        )
        if kind == "Flashbangs":
            value = target.get("FlashDurationAdded")
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
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
                if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
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
        result["impact"] = int(result["enemy_damage"] > 0 or result["enemy_flash_added"] > 0)
    return result, positions


def sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def metrics(y, probabilities, threshold=0.5) -> dict:
    predicted = probabilities >= threshold
    both = len(np.unique(y)) == 2
    return {
        "roc_auc": float(roc_auc_score(y, probabilities)) if both else None,
        "average_precision": float(average_precision_score(y, probabilities)) if both else None,
        "accuracy": float(accuracy_score(y, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(y, predicted)),
        "precision": float(precision_score(y, predicted, zero_division=0)),
        "recall": float(recall_score(y, predicted, zero_division=0)),
        "brier": float(brier_score_loss(y, probabilities)),
        "log_loss": float(log_loss(y, probabilities, labels=[0, 1])),
        "confusion_matrix": confusion_matrix(y, predicted, labels=[0, 1]).tolist(),
    }


def split_indices(features, y, groups, seed, folds, match_dates=None):
    if len(set(groups)) < max(10, folds + 2) or len(np.unique(y)) != 2:
        raise ValueError("Need both labels and at least 10 distinct matches")
    if match_dates is None:
        train, test = next(
            GroupShuffleSplit(
                n_splits=1,
                test_size=0.2,
                random_state=seed,
            ).split(features, y, groups)
        )
        cv = list(
            StratifiedGroupKFold(
                n_splits=folds,
                shuffle=True,
                random_state=seed,
            ).split(train, y[train], groups[train])
        )
        kind = "GroupShuffleSplit by match_id; test_size=0.2"
    else:
        missing = set(groups) - set(match_dates)
        if missing:
            raise ValueError(f"Date CSV lacks {len(missing)} labeled matches")
        times = np.array([match_dates[g] for g in groups])
        unique_times = np.unique(times)
        if len(unique_times) < folds + 3:
            raise ValueError("Too few distinct timestamps for temporal validation")
        cutoff = unique_times[int(0.8 * len(unique_times))]
        train, test = np.flatnonzero(times < cutoff), np.flatnonzero(times >= cutoff)
        train_times = np.unique(times[train])
        cv = [
            (
                np.flatnonzero(np.isin(times[train], train_times[a])),
                np.flatnonzero(np.isin(times[train], train_times[b])),
            )
            for a, b in TimeSeriesSplit(n_splits=folds).split(train_times)
        ]
        kind = "chronological holdout; expanding-window validation; timestamp ties kept together"
        assert times[train].max() < times[test].min()
    if set(groups[train]) & set(groups[test]):
        raise AssertionError("Train/test match leakage")
    for a, b in [(train, test)] + [(train[a], train[b]) for a, b in cv]:
        if not len(a) or not len(b) or min(len(np.unique(y[a])), len(np.unique(y[b]))) < 2:
            raise ValueError(
                "A partition has no observations or only one label; do not retry using test outcomes"
            )
        if set(groups[a]) & set(groups[b]):
            raise AssertionError("Validation match leakage")
    return train, test, cv, kind


def fingerprint(x, y, groups):
    digest = hashlib.sha256(x.tobytes() + y.tobytes())
    for group in groups:
        digest.update(str(group).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def read_analysis(root=ROOT):
    path = Path(root) / "reports/analysis.json"
    return json.loads(path.read_text()) if path.exists() else {}


def save_analysis(section, value, root=ROOT):
    path = Path(root) / "reports/analysis.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    report = read_analysis(root)
    report[section] = value
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)
