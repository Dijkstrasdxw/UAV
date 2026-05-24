"""Draw a thesis-style width-height distribution reference figure.

This script generates a synthetic distribution only for layout/style testing.
Do not describe the output as a real dataset statistic unless it is replaced
with values computed from actual labels.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, PowerNorm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Draw a synthetic width-height reference figure.")
    parser.add_argument("--output", type=Path, default=Path("res") / "synthetic_wh_reference.png")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--bins", type=int, default=52)
    parser.add_argument("--xmax", type=float, default=0.55)
    parser.add_argument("--ymax", type=float, default=0.66)
    return parser.parse_args()


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 10,
            "axes.labelsize": 13,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.linewidth": 0.9,
            "savefig.bbox": "tight",
        }
    )


def build_cmap() -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(
        "reference_blue",
        ["#FFFFFF", "#EEF5FD", "#D7E8F8", "#B7D3F0", "#8CB9E3", "#5B99D3", "#1E6DB7"],
    )


def make_synthetic_boxes(seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)

    main_count = 220_000
    widths = rng.beta(1.8, 3.6, size=main_count) * 0.36
    widths += rng.normal(0.0, 0.012, main_count)
    widths = np.clip(widths, 0.002, 0.34)

    upper = 0.11 + 2.2 * widths - 4.6 * widths**2
    upper = np.clip(upper, 0.07, 0.34)
    lower = np.clip(0.018 + 0.055 * widths + rng.normal(0.0, 0.004, main_count), 0.006, None)
    ratios = rng.beta(1.2, 1.8, size=main_count)
    heights = lower + ratios * (upper - lower)
    heights += rng.normal(0.0, 0.010, main_count)
    heights = np.clip(heights, 0.006, 0.36)

    small_count = 48_000
    small_w = rng.gamma(shape=1.4, scale=0.015, size=small_count)
    small_h = rng.gamma(shape=1.6, scale=0.018, size=small_count)
    small_w = np.clip(small_w, 0.001, 0.12)
    small_h = np.clip(small_h, 0.004, 0.14)

    out_count = 170
    out_w = np.clip(rng.beta(1.5, 2.6, out_count) * 0.52, 0.06, 0.53)
    out_h = np.clip(rng.beta(1.8, 2.2, out_count) * 0.62 + 0.04, 0.05, 0.65)

    widths = np.concatenate([widths, small_w, out_w])
    heights = np.concatenate([heights, small_h, out_h])
    return widths, heights


def plot(widths: np.ndarray, heights: np.ndarray, args: argparse.Namespace) -> None:
    hist, x_edges, y_edges = np.histogram2d(
        widths,
        heights,
        bins=args.bins,
        range=[[0.0, args.xmax], [0.0, args.ymax]],
    )

    masked = np.ma.masked_where(hist.T <= 0, hist.T)
    vmax = np.percentile(hist[hist > 0], 98.8)

    fig, ax = plt.subplots(figsize=(4.4, 3.9))
    mesh = ax.pcolormesh(
        x_edges,
        y_edges,
        masked,
        cmap=build_cmap(),
        norm=PowerNorm(gamma=0.34, vmin=0, vmax=vmax),
        shading="flat",
    )
    mesh.set_edgecolor("face")

    ax.set_xlim(0.0, args.xmax)
    ax.set_ylim(0.0, args.ymax)
    ax.set_xlabel("width")
    ax.set_ylabel("height")
    ax.tick_params(direction="out", length=3.5, width=0.9)

    for spine in ax.spines.values():
        spine.set_linewidth(0.9)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=args.dpi, facecolor="white", pad_inches=0.02)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    setup_style()
    widths, heights = make_synthetic_boxes(args.seed)
    plot(widths, heights, args)
    print(f"Saved figure to: {args.output.resolve()}")


if __name__ == "__main__":
    main()
