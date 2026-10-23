"""Все рисунки PoC: пространство событий, эталоны, противники и проверка ML."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from ml_algorithm import DEFAULT_DATABASE, Team, database, read_document

TEAM_LABELS = {Team.T: "T", Team.CT: "CT"}
COLORS = {10: "#2463a0", 15: "#bf721d"}


def draw_grid(grid: list[dict], extraction: dict, references: list[dict]):
    """Сопоставить стороны на общей шкале плотности без имитации геометрии стен."""
    sizes = np.asarray(extraction["grid_size"])
    cells = {side: Counter() for side in (Team.T, Team.CT)}
    for row in grid:
        if row["point"] == "player":
            cells[row["side"]][tuple(row[k] for k in ("ix", "iy", "iz"))] += row[
                "throws"
            ]
    maximum = max(max(counts.values()) for counts in cells.values())
    normalizer = matplotlib.colors.LogNorm(vmin=1, vmax=maximum)
    bounds = np.asarray(extraction["coordinate_bounds"])
    low = np.floor(bounds[:, 0] / sizes) * sizes
    high = (np.floor(bounds[:, 1] / sizes) + 1) * sizes
    fig = plt.figure(figsize=(14, 8))
    fig.suptitle("Cache: позиции бросков в пространстве X, Y, Z", fontsize=17, y=0.97)
    fig.text(
        0.04,
        0.91,
        f"Все {extraction['matches']:,} матчей · {extraction['counts']['throws']:,} бросков · ячейка 256 × 256 × 64".replace(
            ",", " "
        ),
    )
    for index, (side, counts) in enumerate(cells.items(), start=1):
        ax = fig.add_subplot(1, 2, index, projection="3d")
        centers = (np.asarray(list(counts)) + 0.5) * sizes
        ax.scatter(
            *centers.T,
            c=list(counts.values()),
            s=12,
            marker="s",
            cmap="viridis",
            norm=normalizer,
            depthshade=False,
            linewidths=0,
        )
        for reference in references:
            if reference["side"] in (None, side):
                ax.scatter(
                    *reference["setup"],
                    marker="*",
                    s=65,
                    color="#e79b35",
                    edgecolors="black",
                    linewidths=0.4,
                )
        ax.set(
            xlim=(low[0], high[0]),
            ylim=(low[1], high[1]),
            zlim=(low[2], high[2]),
            xlabel="X",
            ylabel="Y",
            zlabel="Z",
            title=f"{TEAM_LABELS[side]} · код {int(side)}",
        )
        ax.set_box_aspect((1, 0.8, 0.4))
        ax.view_init(elev=27, azim=-65)
        ax.tick_params(labelsize=8)
        ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(1500))
        ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(1000))
        ax.zaxis.set_major_locator(matplotlib.ticker.MultipleLocator(128))
    color_axis = fig.add_axes([0.935, 0.27, 0.012, 0.45])
    fig.colorbar(
        matplotlib.cm.ScalarMappable(norm=normalizer, cmap="viridis"),
        cax=color_axis,
        label="Бросков в ячейке, логарифмическая шкала",
    )
    fig.subplots_adjust(left=0.01, right=0.90, bottom=0.17, top=0.84, wspace=0.02)
    fig.text(
        0.04,
        0.09,
        "Звёзды — внешние позиции подготовки. Высота визуально растянута для читаемости.",
    )
    fig.text(
        0.04,
        0.05,
        "Это пространство событий: стены, коллизии и полная траектория по этим данным не восстанавливаются.",
    )
    return fig


def draw_associations(statistics: list[dict], references: list[dict]):
    flashes = [r for r in references if r["type"] == "Flashbangs"]
    fig, ax = plt.subplots(figsize=(13, 6.8))
    fig.subplots_adjust(left=0.28, right=0.69, top=0.77, bottom=0.23)
    for i, reference in enumerate(flashes):
        for tolerance, offset in ((10, -0.13), (15, 0.13)):
            row = next(
                r
                for r in statistics
                if r["reference"] == reference["id"] and r["tolerance_pct"] == tolerance
            )
            y = i + offset
            interval = row["delta_low"], row["delta_high"]
            if all(v is not None for v in interval):
                ax.plot(
                    np.asarray(interval) * 100,
                    [y, y],
                    color=COLORS[tolerance],
                    linewidth=2,
                )
                ax.scatter(row["delta"] * 100, y, color=COLORS[tolerance], s=38)
            else:
                ax.text(
                    0.03,
                    y,
                    "Интервал не оценён",
                    transform=ax.get_yaxis_transform(),
                    va="center",
                    fontsize=9,
                    color="#6c6c6c",
                )
            note = f"{tolerance}%: {row['within_known']}/{row['outside_known']} бросков; {row['within_matches']}/{row['outside_matches']} матч."
            ax.text(
                1.03,
                y,
                note,
                transform=ax.get_yaxis_transform(),
                va="center",
                fontsize=9,
                color=COLORS[tolerance],
            )
    ax.axvline(0, color="#777777", linestyle="--", linewidth=1)
    ax.set(
        yticks=range(len(flashes)),
        yticklabels=[f"{r['name']}\n{TEAM_LABELS[r['side']]}" for r in flashes],
        xlabel="Разность долей с импактом: внутри допуска − вне допуска, п.п.",
        ylim=(len(flashes) - 0.5, -0.5),
    )
    ax.grid(axis="x", alpha=0.18)
    fig.suptitle(
        "Близость к эталону и зарегистрированное ослепление", fontsize=17, y=0.96
    )
    fig.text(
        0.04,
        0.87,
        "Только броски стоя с известным исходом · 95% интервалы: bootstrap целых матчей",
    )
    fig.text(0.71, 0.80, "Внутри / вне допуска", fontsize=10)
    fig.text(
        0.04,
        0.12,
        "Числа справа относятся к отдельному эталону и фильтрам; исходный охват — 10 000 матчей.",
    )
    fig.text(
        0.04,
        0.07,
        "Для подгруппы менее 5 матчей интервал не строится. Связь не доказывает причинный эффект раскидки.",
    )
    return fig


def draw_enemy_example(examples: dict, references: list[dict]):
    example = examples["prac_cfamtw/3"]
    row = example["throw"]
    reference = next(r for r in references if r["id"] == row["reference"])
    fig, ax = plt.subplots(figsize=(12, 7))
    actual = np.asarray(
        [
            [row[f"{name}_{axis}"] for axis in "xyz"]
            for name in ("player", "start", "end")
        ]
    )
    ax.scatter(
        *actual[:, :2].T, s=[65, 35, 65], color=["#2463a0", "#72a8cc", "#355e3b"]
    )
    ax.scatter(
        *reference["setup"][:2],
        marker="*",
        s=140,
        color="#e79b35",
        label="Эталонная позиция",
    )
    for point, label, offset in zip(
        actual,
        ("Игрок", "Начало полёта", "Конец полёта"),
        ((-20, -40), (20, 25), (-100, 30)),
    ):
        ax.annotate(
            f"{label}\nZ={point[2]:.0f}",
            xy=point[:2],
            xytext=offset,
            textcoords="offset points",
            fontsize=9,
            arrowprops={"arrowstyle": "-", "color": "#777777"},
        )
    # Отрезок связывает наблюдаемые точки и не изображает физическую траекторию.
    ax.plot(
        *actual[1:, :2].T,
        linestyle=":",
        color="#777777",
        label="Связь начальной и конечной точек",
    )
    for index, target in enumerate(example["targets"]):
        point = target["position"]
        ax.scatter(*point[:2], color="#b94b36", marker="x", s=65)
        ax.annotate(
            f"Противник {index + 1}: +{target['value']:.2f} с\nZ={point[2]:.0f}",
            xy=point[:2],
            xytext=(15, -30),
            textcoords="offset points",
            fontsize=9,
            color="#943e2e",
        )
    ax.set(xlabel="X", ylabel="Y", title="Проекция XY; высоты Z указаны в подписях")
    ax.set_aspect("equal", adjustable="datalim")
    ax.margins(x=0.18, y=0.4)
    ax.grid(alpha=0.18)
    ax.legend(loc="upper left", bbox_to_anchor=(0, 1.01), fontsize=8)
    fig.suptitle(
        "Положение противников при зарегистрированном ослеплении", fontsize=16, y=0.96
    )
    fig.text(
        0.04,
        0.88,
        f"Одна иллюстрация · CT · раунд {row['round'] + 1} · матч {row['match_id']}",
    )
    fig.text(
        0.04,
        0.09,
        f"Суммарно добавлено {row['enemy_flash_added']:.2f} с ослепления. Цели без записи эффекта не показаны.",
    )
    fig.subplots_adjust(left=0.10, right=0.94, top=0.77, bottom=0.22)
    return fig


def draw_model(result: dict, predictions: list[dict], title: str):
    from sklearn.calibration import calibration_curve
    from sklearn.metrics import roc_curve

    test = [r for r in predictions if r["partition"] == "test"]
    y = np.asarray([r["label"] for r in test])
    probability = np.asarray([r["probability"] for r in test])
    false_positive, true_positive, _ = roc_curve(y, probability)
    observed, predicted = calibration_curve(
        y, probability, n_bins=8, strategy="quantile"
    )
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


def draw_correlations(outcomes: dict):
    rows = [r for r in outcomes["correlations"] if r["pearson_ci95"] is not None]
    if not rows:
        return None
    labels = {
        "player_kpd": "КПД",
        "utility_rate": "Доля импакта в пуле",
        "round_won": "Победа в раунде",
    }
    fig, ax = plt.subplots(figsize=(11, 5))
    for i, row in enumerate(rows):
        ax.plot(row["pearson_ci95"], [i, i], color="#2463a0")
        ax.scatter(row["pearson"], i, color="#2463a0")
    ax.axvline(0, linestyle="--", color="#888888")
    ax.set(
        xlim=(-1, 1),
        yticks=range(len(rows)),
        yticklabels=[f"{labels[r['x']]} ↔ {labels[r['y']]}" for r in rows],
        xlabel="Pearson; 95% интервал с кластеризацией по матчу",
    )
    fig.suptitle("Связи с внешним КПД и исходами раундов")
    fig.text(
        0.04,
        0.05,
        "Только покрытые пулом игроки-раунды с известным эффектом. Ретроспективные наблюдения.",
    )
    fig.subplots_adjust(left=0.42, bottom=0.23, top=0.85)
    return fig


def render(path: Path, output: Path | None) -> None:
    if output is not None:
        plt.switch_backend("Agg")
        output.mkdir(parents=True, exist_ok=True)
    with database(path) as connection:
        grid = [dict(row) for row in connection.execute("SELECT * FROM spatial_grid")]
        documents = {
            key: read_document(connection, key)
            for key in (
                "extraction",
                "references",
                "examples",
                "associations",
                "flash_model",
                "flash_predictions",
                "outcomes",
                "round_predictions",
            )
        }
    with plt.rc_context(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    ):
        references = documents["references"]["references"]
        builders = [
            (
                "spatial_grid",
                lambda: draw_grid(grid, documents["extraction"], references),
            ),
            (
                "reference_associations",
                lambda: draw_associations(documents["associations"], references),
            ),
            (
                "enemy_example",
                lambda: draw_enemy_example(documents["examples"], references),
            ),
        ]
        if documents["flash_model"]["status"] == "trained":
            builders.append(
                (
                    "flash_model",
                    lambda: draw_model(
                        documents["flash_model"],
                        documents["flash_predictions"],
                        "Флешки из пула: эффект по признакам момента броска",
                    ),
                )
            )
        outcomes = documents["outcomes"]
        if outcomes["status"] == "context_loaded":
            builders.append(("kpd_correlations", lambda: draw_correlations(outcomes)))
            if outcomes["round_model"]["status"] == "trained":
                builders.append(
                    (
                        "round_model",
                        lambda: draw_model(
                            outcomes["round_model"],
                            documents["round_predictions"],
                            "Победа: ретроспективная модель по данным раунда",
                        ),
                    )
                )
        else:
            print(
                "КПД и победы: график не построен — в исходнике отсутствуют нужные данные."
            )
        for name, build in builders:
            figure = build()
            if figure is None:
                continue
            if output is None:
                plt.show()
            else:
                target = output / f"{name}.png"
                figure.savefig(target, dpi=150, facecolor="white")
                print(target.resolve())
            plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATABASE)
    args = parser.parse_args()
    render(args.data, args.output_dir)


if __name__ == "__main__":
    main()
