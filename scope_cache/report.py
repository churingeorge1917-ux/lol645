"""Бэк для utility_analysis.ipynb"""

from __future__ import annotations
import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq
from matplotlib.collections import LineCollection
from matplotlib.patches import Ellipse
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scope_cache import ROOT, ASSETS
from scope_cache.geometry import (
    MapGeometry,
    SMOKE_PARAMETERS,
    read_mesh,
    segment_distances,
    smoke_chords,
)

STYLE = {
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
}


def draw_availability(source):
    kinds = ["Smokes", "Flashbangs", "HEs", "Fires"]
    labels = ["Смок", "Флешка", "HE", "Молотов / зажигательная"]
    known = np.array([source["counts"][kind + "/known"] for kind in kinds])
    positive = np.array([source["counts"][kind + "/positive"] for kind in kinds])
    total = np.array([source["counts"][kind] for kind in kinds])
    fig, ax = plt.subplots(figsize=(12, 6))
    left = np.zeros(4)
    for values, label, color, hatch in [
        (positive, "Зарегистрирован эффект", "#35658c", None),
        (known - positive, "Известный нулевой эффект", "#c99135", "//"),
        (total - known, "Исход неизвестен / неприменим", "#d2d5d8", None),
    ]:
        ax.barh(
            labels,
            values / 1000,
            left=left / 1000,
            label=label,
            color=color,
            hatch=hatch,
            edgecolor="white",
            height=0.62,
        )
        left += values
    for i, n in enumerate(total):
        ax.text(n / 1000 + 8, i, f"{n:,}".replace(",", " "), va="center")
    ax.invert_yaxis()
    ax.set(
        xlabel="Бросков, тыс.",
        xlim=(0, 850),
        title="Доступность наблюдаемого эффекта по типу гранаты",
    )
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.17),
        ncol=3,
        frameon=False,
        fontsize=10,
    )
    ax.grid(axis="x", alpha=0.15)
    ax.set_axisbelow(True)
    fig.subplots_adjust(left=0.20, right=0.97, top=0.84, bottom=0.27)
    return fig


def radar(ax, assets=ASSETS, alpha=0.65):
    transform = json.loads((assets / "radar_transform.json").read_text())
    picture = plt.imread(assets / "de_cache.png")
    height, width = picture.shape[:2]
    x, y, scale = transform["pos_x"], transform["pos_y"], transform["scale"]
    ax.imshow(
        picture,
        extent=[x, x + width * scale, y - height * scale, y],
        origin="upper",
        alpha=alpha,
    )
    ax.set(xlabel="X, Hammer units", ylabel="Y, Hammer units")
    ax.set_aspect("equal")


def draw_map():
    vertices, faces = read_mesh()
    geometry = MapGeometry()
    fig = plt.figure(figsize=(14, 7))
    ax = fig.add_subplot(121)
    radar(ax)
    ax.add_collection(
        LineCollection(
            np.stack([geometry.a[::35, :2], geometry.b[::35, :2]], axis=1),
            colors="#2463a0",
            alpha=0.15,
            linewidths=0.5,
        )
    )
    ax.scatter(*geometry.a[::40, :2].T, s=4, color="#aa5127", label="Выборка точек обзора")
    ax.set_title("Радар и выборка статических линий обзора")
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
    ax3 = fig.add_subplot(122, projection="3d")
    sampled = vertices[faces[::20]]
    ax3.add_collection3d(
        Poly3DCollection(
            sampled,
            facecolors="#9cabb8",
            edgecolors="#5e7181",
            linewidths=0.05,
            alpha=0.35,
        )
    )
    low, high = vertices.min(0), vertices.max(0)
    ax3.set(
        xlim=(low[0], high[0]),
        ylim=(low[1], high[1]),
        zlim=(low[2], high[2]),
        xlabel="X",
        ylabel="Y",
        zlabel="Z",
        title="Геометрия карты",
    )
    ax3.set_box_aspect((1, 1, 0.4))
    ax3.view_init(elev=33, azim=-65)
    ax3.tick_params(labelsize=8)
    fig.suptitle(
        "Cache",
        fontsize=16,
        y=0.98,
    )
    fig.subplots_adjust(left=0.07, right=0.96, top=0.85, bottom=0.16, wspace=0.12)
    return fig


def draw_smoke():
    scores = pq.read_table(
        ROOT / "data" / "grenade_scores.parquet",
        filters=[("type", "=", "Smokes")],
        columns=[
            "potential_smoke_coverage",
            "smoke_coverage_radius96",
            "smoke_coverage_radius192",
            "candidate_static_sightlines",
        ],
    )
    endpoints = pq.read_table(
        ROOT / "data" / "events.parquet",
        filters=[("type", "=", "Smokes")],
        columns=["end_x", "end_y", "end_z"],
    )
    coverage = scores["potential_smoke_coverage"].to_numpy()
    candidates = scores["candidate_static_sightlines"].to_numpy()
    finite = np.isfinite(coverage)
    target = np.median(coverage[finite & (coverage > 0)])
    eligible = np.flatnonzero(finite & (coverage > 0) & (candidates >= 100))
    index = eligible[np.argmin(np.abs(coverage[eligible] - target))]
    endpoint = np.array([endpoints["end_" + axis][int(index)].as_py() for axis in "xyz"])
    center = np.round(endpoint / 16) * 16 + [0, 0, SMOKE_PARAMETERS["center_height"]]
    geometry = MapGeometry()
    local = segment_distances(geometry.a, geometry.b, center) <= SMOKE_PARAMETERS["context_radius"]
    a, b = geometry.a[local], geometry.b[local]
    blocked = smoke_chords(a, b, center) >= SMOKE_PARAMETERS["minimum_chord"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.8))
    ax = axes[0]
    radar(ax, alpha=0.35)
    for mask, color, label, width in [
        (~blocked, "#818b93", "Не пересекают объём", 0.5),
        (blocked, "#a84c2b", "Пересекают объём", 0.8),
    ]:
        indices = np.flatnonzero(mask)
        shown = (
            indices[np.linspace(0, len(indices) - 1, min(130, len(indices)), dtype=int)]
            if len(indices)
            else indices
        )
        ax.add_collection(
            LineCollection(
                np.stack([a[shown, :2], b[shown, :2]], axis=1),
                colors=color,
                linewidths=width,
                alpha=0.4,
                label=label,
            )
        )
    for radius, style in [(96, ":"), (144, "-"), (192, "--")]:
        ax.add_patch(
            Ellipse(
                center[:2],
                2 * radius,
                2 * radius,
                fill=False,
                edgecolor="#102f49",
                linestyle=style,
                linewidth=1.5,
                zorder=5,
            )
        )
    ax.set(
        xlim=(center[0] - 900, center[0] + 900),
        ylim=(center[1] - 900, center[1] + 900),
        title=f"Смок: {int(blocked.sum())}/{len(a)} линий; рисунок прорежен(используется лишь часть геометрии карты)",
    )
    ax.legend(loc="upper left", fontsize=9)
    ax = axes[1]
    for key, label, color, style in [
        ("smoke_coverage_radius96", "Радиус XY 96", "#a57b32", ":"),
        ("potential_smoke_coverage", "Радиус XY 144", "#285e88", "-"),
        ("smoke_coverage_radius192", "Радиус XY 192", "#8e4a6f", "--"),
    ]:
        values = scores[key].to_numpy()
        values = np.sort(values[np.isfinite(values)])
        quantiles = np.linspace(0, 1, 1001)
        ax.plot(
            np.quantile(values, quantiles) * 100,
            quantiles * 100,
            label=label,
            color=color,
            linestyle=style,
        )
    ax.set(
        xlabel="Доля перекрытых статических линий, %",
        ylabel="Доля бросков с оценкой не выше, %",
        xlim=(0, 100),
        ylim=(0, 100),
        title="Чувствительность к предполагаемому размеру дыма",
    )
    ax.grid(alpha=0.15)
    ax.legend(loc="lower right", fontsize=10)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.84, bottom=0.24, wspace=0.25)
    fig.suptitle(
        "Смок: геометрическое перекрытие линий обзора",
        fontsize=16,
        y=0.98,
    )
    fig.text(
        0.06,
        0.08,
        f"729 947 смоков; {int((~finite).sum()):,} без подходящих линий. Радиусы — сценарии чувствительности".replace(
            ",", " "
        ),
        fontsize=10,
    )
    fig.text(
        0.06,
        0.035,
        "Высота объёма и его рост не калиброваны по движку. В исходнике нет синхронных взглядов и позиций противников.",
        fontsize=10,
    )
    return fig


def draw_selection(model):
    labels = {
        "setup_catboost": "Состояние бросающего · CatBoost",
        "release_logistic": "Контекст броска · Logistic regression",
        "release_tree": "Контекст броска · Decision tree",
        "release_hist": "Контекст броска · HistGradientBoosting",
        "release_catboost": "Контекст броска · CatBoost",
        "release_rbf": "Контекст броска · RBF + Logistic regression",
    }
    candidates = model["selection"]["candidates"]
    ordered = sorted(candidates, key=lambda k: candidates[k]["oof"]["log_loss"])
    values = [candidates[k]["oof"]["log_loss"] for k in ordered]
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, (key, value) in enumerate(zip(ordered, values)):
        color = "#a84c2b" if key == model["selected_model"] else "#35658c"
        ax.scatter(value, i, color=color, s=90, zorder=3)
        ax.text(value + 0.002, i, f"{value:.5f}", va="center", color=color)
    ax.set(
        yticks=range(len(ordered)),
        yticklabels=[labels[k] for k in ordered],
        xlabel="Train OOF log loss",
        xlim=(min(values) - 0.012, max(values) + 0.026),
        title="Выбор по пяти разбиениям обучающих матчей",
    )
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.18)
    fig.subplots_adjust(left=0.35, right=0.96, top=0.84, bottom=0.21)
    fig.text(
        0.06,
        0.06,
        "Фиксированные конфигурации",
        fontsize=10,
    )
    return fig


def draw_sampling(diagnostics, path=ROOT / "reports/smoke_sensitivity.npz"):
    with np.load(path, allow_pickle=False) as source:
        reference = source["coverage_128"][:, 1]
        alternate = {key: source[f"coverage_{key}"][:, 1] for key in [64, 256]}
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.6))
    for ax, setting in zip(axes, diagnostics["sampling_sensitivity"]):
        step = setting["xy_bin_units"]
        values = alternate[step]
        valid = np.isfinite(reference) & np.isfinite(values)
        ax.scatter(reference[valid], values[valid], s=8, alpha=0.30, color="#35658c")
        ax.plot([0, 1], [0, 1], "--", color="#8b8e91", linewidth=1)
        ax.set(
            xlim=(0, 1),
            ylim=(0, 1),
            xlabel="Доля перекрытых линий",
            ylabel="Доля перекрытых линий",
            title=f"Шаг {step} · Spearman {setting['spearman_rank_correlation']:.3f} · n={valid.sum()}",
        )
        ax.set_aspect("equal")
        ax.grid(alpha=0.15)
    fig.suptitle("Смок: чувствительность к выборке точек обзора", fontsize=16, y=0.97)
    fig.subplots_adjust(left=0.07, right=0.97, top=0.81, bottom=0.22, wspace=0.25)
    fig.text(
        0.06,
        0.08,
        "2 000 случайных ячеек точек раскрытия смока с равными весами; радиус XY = 144. Нулевой знаменатель исключён попарно.",
        fontsize=10,
    )
    fig.text(
        0.06,
        0.035,
        "Стабильность расчёта не подтверждает фактическое перекрытие взгляда. Плотность и форма дыма не калиброваны по игре.",
        fontsize=10,
    )
    return fig


def selection_table(model):
    rows = ["| Кандидат | Train OOF log loss | Train OOF ROC-AUC |", "|---|---:|---:|"]
    for name, candidate in sorted(
        model["selection"]["candidates"].items(),
        key=lambda item: item[1]["oof"]["log_loss"],
    ):
        m = candidate["oof"]
        rows.append(f"| {name} | {m['log_loss']:.6f} | {m['roc_auc']:.4f} |")
    return "\n".join(rows)


def uncertainty_table(diagnostics):
    rows = ["| Метрика test | Оценка | 95% интервал по матчам |", "|---|---:|---:|"]
    for name, row in diagnostics["uncertainty"]["metrics"].items():
        lo, hi = row["ci95"]
        rows.append(f"| {name} | {row['estimate']:.4f} | {lo:.4f}–{hi:.4f} |")
    return "\n".join(rows)


def draw_model(result: dict, predictions: list[dict], title: str):
    from sklearn.calibration import calibration_curve
    from sklearn.metrics import roc_curve

    test = [r for r in predictions if r["partition"] == "test"]
    y = np.asarray([r["label"] for r in test])
    probability = np.asarray([r["probability"] for r in test])
    false_positive, true_positive, _ = roc_curve(y, probability)
    observed, predicted = calibration_curve(y, probability, n_bins=8, strategy="quantile")
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].plot(
        false_positive,
        true_positive,
        color="#2463a0",
        label=f"ROC-AUC = {result['test']['metrics']['roc_auc']:.3f}",
    )
    axes[0].plot([0, 1], [0, 1], linestyle="--", color="#888888")
    axes[0].set(
        xlabel="Доля ложноположительных",
        ylabel="Доля истинноположительных",
        title="Разделение классов",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    axes[0].legend(loc="lower right")
    axes[1].plot(predicted, observed, marker="o", color="#bf721d")
    axes[1].plot([0, 1], [0, 1], linestyle="--", color="#888888")
    axes[1].set(
        xlabel="Предсказанная вероятность",
        ylabel="Наблюдаемая доля",
        title="Калибровка: 8 групп равного размера",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    for ax in axes:
        ax.grid(alpha=0.18)
    fig.suptitle(title, fontsize=16, y=0.97)
    fig.text(
        0.07,
        0.87,
        f"Test: {result['test']['rows']:,} наблюдений / {result['test']['matches']:,} матчей · матчи train и test не пересекаются".replace(
            ",", " "
        ),
    )
    fig.text(
        0.07,
        0.05,
        f"Brier: модель {result['test']['metrics']['brier']:.3f}; постоянная вероятность train {result['baseline']['brier']:.3f} (меньше — лучше).",
    )
    fig.subplots_adjust(left=0.07, right=0.97, bottom=0.19, top=0.78, wspace=0.25)
    return fig


def execute_in_process(notebook):
    import nbformat
    from IPython.core.interactiveshell import InteractiveShell
    from IPython.utils.capture import capture_output

    class InlineShell(InteractiveShell):
        def enable_gui(self, gui=None):
            if gui is not None:
                raise ValueError("In-process execution supports only inline Matplotlib")

    shell = InlineShell.instance()
    previous = Path.cwd()
    os.chdir(ROOT)
    try:
        for cell in notebook.cells:
            if cell.cell_type != "code":
                continue
            cell.execution_count = shell.execution_count
            with capture_output(stdout=True, stderr=True, display=True) as captured:
                result = shell.run_cell(cell.source, store_history=True)
            if result.error_before_exec or result.error_in_exec:
                raise RuntimeError(f"Cell {cell.execution_count} failed") from (
                    result.error_before_exec or result.error_in_exec
                )
            cell.outputs = []
            for name, value in (
                ("stdout", captured.stdout),
                ("stderr", captured.stderr),
            ):
                if value:
                    cell.outputs.append(nbformat.v4.new_output("stream", name=name, text=value))
            cell.outputs.extend(
                nbformat.v4.new_output("display_data", data=output.data, metadata=output.metadata)
                for output in captured.outputs
            )
    finally:
        os.chdir(previous)
    notebook.metadata["execution_method"] = (
        "in-process IPython; real cell execution and captured rich displays"
    )


def execute_report(output=None):
    import nbformat

    path = ROOT / "utility_analysis.ipynb"
    notebook = nbformat.read(path, as_version=4)
    execute_in_process(notebook)
    nbformat.validate(notebook)
    images = sum(
        "image/png" in o.get("data", {}) for c in notebook.cells for o in c.get("outputs", [])
    )
    output = Path(output) if output is not None else path
    output.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook, output)
    print(f"Выполнено: {len(notebook.cells)} ячеек, {images} графиков; {output}")
