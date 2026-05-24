"""Plot per-category AP comparison between baseline and improved model.

Usage:
    python tools/plot_category_ap_comparison.py
    python tools/plot_category_ap_comparison.py --output res/category_ap_comparison.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot category-wise AP comparison.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("res") / "category_ap_comparison.png",
        help="Output image path.",
    )
    parser.add_argument("--dpi", type=int, default=300, help="Saved image DPI.")
    parser.add_argument(
        "--title",
        type=str,
        default="Category-wise AP Comparison on VisDrone2019",
        help="Figure title.",
    )
    return parser.parse_args()


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Microsoft YaHei", "SimHei", "Arial", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.linestyle": "--",
            "grid.alpha": 0.35,
            "savefig.bbox": "tight",
        }
    )


def add_value_labels(ax: plt.Axes, bars: list[plt.Rectangle], values: np.ndarray) -> None:
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.7,
            f"{value:.1f}",
            ha="center",
            va="bottom",
            fontsize=9,
            color="#2B2B2B",
        )


def main() -> None:
    args = parse_args()
    setup_style()

    categories = [
        "行人",
        "人群",
        "自行车",
        "汽车",
        "箱式货车",
        "卡车",
        "三轮车",
        "遮阳三轮",
        "公交车",
        "摩托车",
        "全部",
    ]
    baseline = np.array([30.1, 16.8, 12.6, 74.2, 40.2, 43.7, 22.0, 18.9, 59.8, 32.6, 35.1], dtype=float)
    ours = np.array([38.5, 24.4, 15.3, 79.5, 45.6, 47.2, 24.8, 25.6, 62.8, 38.9, 40.3], dtype=float)

    display_labels = [
        "行人",
        "人群",
        "自行车",
        "汽车",
        "箱式\n货车",
        "卡车",
        "三轮车",
        "遮阳\n三轮",
        "公交车",
        "摩托车",
        "全部",
    ]

    x = np.arange(len(categories))
    width = 0.36

    fig, ax = plt.subplots(figsize=(12.8, 5.6))
    bars1 = ax.bar(
        x - width / 2,
        baseline,
        width=width,
        color="#87CEEB",
        edgecolor="#5B9BD5",
        linewidth=1.0,
        label="原始模型",
        zorder=3,
    )
    bars2 = ax.bar(
        x + width / 2,
        ours,
        width=width,
        color="#90EE90",
        edgecolor="#70AD47",
        linewidth=1.0,
        label="改进模型",
        zorder=3,
    )

    add_value_labels(ax, list(bars1), baseline)
    add_value_labels(ax, list(bars2), ours)
    ax.set_title(args.title, pad=12)
    ax.set_ylabel("AP50 (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(display_labels)
    ax.set_ylim(0, max(ours.max(), baseline.max()) + 12)
    ax.grid(axis="y", zorder=0)
    ax.legend(loc="upper left", frameon=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=args.dpi, facecolor="white")
    plt.close(fig)
    print(f"Saved figure to: {args.output.resolve()}")


if __name__ == "__main__":
    main()
