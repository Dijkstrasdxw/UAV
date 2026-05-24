"""Plot common activation functions as separate thesis figures.

Usage:
    python tools/plot_activation_functions.py
    python tools/plot_activation_functions.py --output-dir res/activation_functions
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def tanh_fn(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


def leaky_relu(x: np.ndarray, alpha: float = 0.01) -> np.ndarray:
    return np.where(x > 0.0, x, alpha * x)


def silu(x: np.ndarray) -> np.ndarray:
    return x * sigmoid(x)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot common activation functions.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("res") / "activation_functions",
        help="Output directory for individual activation plots.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Saved image DPI.",
    )
    parser.add_argument(
        "--xmin",
        type=float,
        default=-10.0,
        help="Minimum x value.",
    )
    parser.add_argument(
        "--xmax",
        type=float,
        default=10.0,
        help="Maximum x value.",
    )
    parser.add_argument(
        "--points",
        type=int,
        default=1000,
        help="Number of sampled points.",
    )
    return parser.parse_args()


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.linewidth": 1.0,
            "grid.linestyle": "--",
            "grid.linewidth": 0.6,
            "grid.alpha": 0.5,
            "savefig.bbox": "tight",
        }
    )


def add_common_style(ax: plt.Axes, title: str, ylabel: str) -> None:
    ax.set_title(title, pad=8)
    ax.set_xlabel("x")
    ax.set_ylabel(ylabel)
    ax.grid(True)
    ax.axhline(0.0, color="#9A9A9A", linewidth=0.9, linestyle="--", zorder=0)
    ax.axvline(0.0, color="#9A9A9A", linewidth=0.9, linestyle="--", zorder=0)


def main() -> None:
    args = parse_args()
    setup_style()

    x = np.linspace(args.xmin, args.xmax, args.points)
    plots = [
        {
            "title": "Sigmoid",
            "ylabel": "f(x)",
            "func": sigmoid,
            "color": "#2F5597",
            "formula": r"$f(x)=\frac{1}{1+e^{-x}}$",
            "filename": "sigmoid.png",
            "ylim": (-0.05, 1.05),
        },
        {
            "title": "Tanh",
            "ylabel": "f(x)",
            "func": tanh_fn,
            "color": "#2E8B57",
            "formula": r"$f(x)=\tanh(x)$",
            "filename": "tanh.png",
            "ylim": (-1.05, 1.05),
        },
        {
            "title": "ReLU",
            "ylabel": "f(x)",
            "func": relu,
            "color": "#C0504D",
            "formula": r"$f(x)=\max(0,x)$",
            "filename": "relu.png",
            "ylim": (-0.5, 10.5),
        },
        {
            "title": "Leaky ReLU",
            "ylabel": "f(x)",
            "func": leaky_relu,
            "color": "#7F3F98",
            "formula": r"$f(x)=\max(\alpha x,x),\ \alpha=0.01$",
            "filename": "leaky_relu.png",
            "ylim": (-0.5, 10.5),
        },
        {
            "title": "SiLU",
            "ylabel": "f(x)",
            "func": silu,
            "color": "#00A6D6",
            "formula": r"$f(x)=x\cdot\sigma(x)$",
            "filename": "silu.png",
            "ylim": (-0.5, 10.5),
        },
    ]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for plot in plots:
        fig, ax = plt.subplots(figsize=(4.8, 3.8), constrained_layout=True)
        y = plot["func"](x)
        ax.plot(x, y, color=plot["color"], linewidth=2.0, label=plot["formula"])
        ax.set_xlim(args.xmin, args.xmax)
        ax.set_ylim(*plot["ylim"])
        add_common_style(ax, plot["title"], plot["ylabel"])
        ax.legend(loc="upper left", frameon=True, framealpha=0.9, borderpad=0.3)
        output_path = args.output_dir / plot["filename"]
        fig.savefig(output_path, dpi=args.dpi)
        plt.close(fig)
        print(f"Saved figure to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
