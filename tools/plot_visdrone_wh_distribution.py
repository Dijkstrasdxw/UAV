"""Plot VisDrone width-height distribution for thesis figures.

This script draws a standalone width-height distribution figure from YOLO-format
labels. The visual style is tuned to be closer to common thesis figures than
Ultralytics' default four-panel `labels.jpg`.

Usage:
    python tools/plot_visdrone_wh_distribution.py
    python tools/plot_visdrone_wh_distribution.py --bins 65 --mode hist2d
    python tools/plot_visdrone_wh_distribution.py --mode paper_scatter
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, PowerNorm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot VisDrone width-height distribution.")
    parser.add_argument(
        "--labels-root",
        type=Path,
        default=Path("VisDrone") / "labels",
        help="Root directory containing train/val/test YOLO labels.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "test"],
        help="Dataset splits to include.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("res") / "visdrone_wh_distribution.png",
        help="Output image path.",
    )
    parser.add_argument("--dpi", type=int, default=300, help="Saved image DPI.")
    parser.add_argument("--bins", type=int, default=60, help="Number of 2D bins.")
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.32,
        help="Power-law normalization gamma used in hist2d mode.",
    )
    parser.add_argument(
        "--vmax-percentile",
        type=float,
        default=99.0,
        help="Upper percentile used to clip dense bins in hist2d mode.",
    )
    parser.add_argument(
        "--mode",
        choices=["hist2d", "scatter", "paper_scatter"],
        default="paper_scatter",
        help="Plot mode. 'paper_scatter' is tuned to resemble thesis-style square-point figures.",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=180000,
        help="Maximum number of points used in scatter modes.",
    )
    parser.add_argument("--xmax", type=float, default=0.55, help="Maximum width shown.")
    parser.add_argument("--ymax", type=float, default=0.65, help="Maximum height shown.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed used for subsampling.")
    return parser.parse_args()


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Times New Roman", "Arial", "DejaVu Sans"],
            "font.size": 12,
            "axes.labelsize": 13,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "savefig.bbox": "tight",
            "axes.linewidth": 1.1,
        }
    )


def load_wh_from_yolo_labels(labels_root: Path, splits: list[str]) -> tuple[np.ndarray, np.ndarray]:
    widths: list[float] = []
    heights: list[float] = []

    for split in splits:
        split_dir = labels_root / split
        if not split_dir.exists():
            continue
        for txt_file in split_dir.rglob("*.txt"):
            with txt_file.open("r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    try:
                        widths.append(float(parts[3]))
                        heights.append(float(parts[4]))
                    except ValueError:
                        continue

    if not widths:
        raise FileNotFoundError(f"No valid labels found under: {labels_root}")

    return np.asarray(widths, dtype=np.float32), np.asarray(heights, dtype=np.float32)


def build_blue_cmap() -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(
        "paper_blue",
        ["#FFFFFF", "#EAF3FE", "#CFE2F8", "#A9C9EE", "#7FB1E7", "#4C8ED6", "#165FAF"],
    )


def plot_hist2d(widths: np.ndarray, heights: np.ndarray, args: argparse.Namespace) -> None:
    fig, ax = plt.subplots(figsize=(5.4, 4.3))
    cmap = build_blue_cmap()
    hist, x_edges, y_edges = np.histogram2d(
        widths,
        heights,
        bins=args.bins,
        range=[[0.0, args.xmax], [0.0, args.ymax]],
    )
    vmax = np.percentile(hist[hist > 0], args.vmax_percentile) if np.any(hist > 0) else 1.0
    hist = np.ma.masked_where(hist.T <= 0, hist.T)
    mesh = ax.pcolormesh(
        x_edges,
        y_edges,
        hist,
        cmap=cmap,
        norm=PowerNorm(gamma=args.gamma, vmin=0, vmax=vmax),
        shading="flat",
    )

    ax.set_xlim(0.0, args.xmax)
    ax.set_ylim(0.0, args.ymax)
    ax.set_xlabel("width")
    ax.set_ylabel("height")
    mesh.set_edgecolor("face")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=args.dpi, facecolor="white")
    plt.close(fig)


def downsample_points(widths: np.ndarray, heights: np.ndarray, sample: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    if len(widths) > sample:
        indices = rng.choice(len(widths), size=sample, replace=False)
        widths = widths[indices]
        heights = heights[indices]
    return widths, heights


def plot_scatter(widths: np.ndarray, heights: np.ndarray, args: argparse.Namespace) -> None:
    widths, heights = downsample_points(widths, heights, args.sample, args.seed)

    fig, ax = plt.subplots(figsize=(5.4, 4.3))
    ax.scatter(
        widths,
        heights,
        s=18,
        marker="s",
        c="#AFCDEE",
        alpha=0.45,
        edgecolors="none",
    )
    ax.set_xlim(0.0, args.xmax)
    ax.set_ylim(0.0, args.ymax)
    ax.set_xlabel("width")
    ax.set_ylabel("height")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=args.dpi, facecolor="white")
    plt.close(fig)


def plot_paper_scatter(widths: np.ndarray, heights: np.ndarray, args: argparse.Namespace) -> None:
    keep = (widths <= args.xmax) & (heights <= args.ymax)
    widths = widths[keep]
    heights = heights[keep]

    fig, ax = plt.subplots(figsize=(5.0, 4.9))
    hist, x_edges, y_edges = np.histogram2d(
        widths,
        heights,
        bins=args.bins,
        range=[[0.0, args.xmax], [0.0, args.ymax]],
    )
    vmax = np.percentile(hist[hist > 0], 99.0) if np.any(hist > 0) else 1.0
    hist = np.ma.masked_where(hist.T <= 0, hist.T)
    mesh = ax.pcolormesh(
        x_edges,
        y_edges,
        hist,
        cmap=build_blue_cmap(),
        norm=PowerNorm(gamma=0.42, vmin=0, vmax=vmax),
        shading="flat",
        alpha=1.0,
    )
    mesh.set_edgecolor("face")

    ax.set_xlim(0.0, args.xmax)
    ax.set_ylim(0.0, args.ymax)
    ax.set_xlabel("width")
    ax.set_ylabel("height")
    ax.tick_params(direction="out", length=4.5, width=1.0)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.1)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=args.dpi, facecolor="white")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    setup_style()
    widths, heights = load_wh_from_yolo_labels(args.labels_root, args.splits)

    if args.mode == "hist2d":
        plot_hist2d(widths, heights, args)
    elif args.mode == "scatter":
        plot_scatter(widths, heights, args)
    else:
        plot_paper_scatter(widths, heights, args)

    print(f"Saved figure to: {args.output.resolve()}")
    print(f"Loaded boxes: {len(widths)}")
    print(f"Average width: {widths.mean():.4f}")
    print(f"Average height: {heights.mean():.4f}")


if __name__ == "__main__":
    main()
