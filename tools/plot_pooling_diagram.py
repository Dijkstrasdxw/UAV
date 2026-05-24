"""Generate a pooling layer schematic figure for thesis use."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrow, Rectangle
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot a pooling layer schematic figure.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("res") / "pooling_diagram.png",
        help="Output image path.",
    )
    parser.add_argument("--dpi", type=int, default=300, help="Saved image DPI.")
    return parser.parse_args()


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Arial Unicode MS", "DejaVu Sans"],
            "font.size": 14,
            "axes.unicode_minus": False,
            "savefig.bbox": "tight",
        }
    )


def draw_grid(
    ax: plt.Axes,
    data: np.ndarray,
    x0: float,
    y0: float,
    cell: float,
    highlight: tuple[int, int, int, int] | None = None,
    fill_color: str = "#8FB1E3",
    border_color: str = "#7AA1E6",
    dashed: bool = False,
) -> None:
    rows, cols = data.shape
    for r in range(rows):
        for c in range(cols):
            x = x0 + c * cell
            y = y0 + (rows - 1 - r) * cell
            facecolor = "white"
            if highlight is not None:
                hr, hc, h, w = highlight
                if hr <= r < hr + h and hc <= c < hc + w:
                    facecolor = fill_color
            rect = Rectangle((x, y), cell, cell, linewidth=1.5, edgecolor="black", facecolor=facecolor)
            ax.add_patch(rect)
            ax.text(x + cell / 2, y + cell / 2, str(int(data[r, c])), ha="center", va="center", fontsize=17)

    outline_style = "--" if dashed else "-"
    outline = Rectangle(
        (x0, y0),
        cols * cell,
        rows * cell,
        linewidth=1.6,
        edgecolor="black",
        facecolor="none",
        linestyle=outline_style,
    )
    ax.add_patch(outline)

    if highlight is not None:
        hr, hc, h, w = highlight
        hx = x0 + hc * cell - 0.12
        hy = y0 + (rows - hr - h) * cell - 0.12
        hrect = Rectangle(
            (hx, hy),
            w * cell + 0.24,
            h * cell + 0.24,
            linewidth=2.0,
            edgecolor=border_color,
            facecolor="none",
            linestyle="--" if dashed else "-",
        )
        ax.add_patch(hrect)


def main() -> None:
    args = parse_args()
    setup_style()

    input_data = np.array(
        [
            [3, 2, 5, 7],
            [1, 4, 2, 6],
            [8, 1, 3, 2],
            [2, 5, 4, 1],
        ]
    )

    max_out = np.array([[4, 7], [8, 4]])
    avg_out = np.array([[2, 5], [4, 2]])

    fig, ax = plt.subplots(figsize=(12, 5.6))
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 7)
    ax.axis("off")

    cell = 1.1
    input_x, input_y = 0.8, 1.0
    out_x, top_y, bottom_y = 10.0, 4.0, 1.0

    highlight = (0, 0, 2, 2)

    draw_grid(ax, input_data, input_x, input_y, cell, highlight=highlight)
    draw_grid(ax, max_out, out_x, top_y, cell, highlight=(0, 0, 1, 1))
    draw_grid(ax, avg_out, out_x, bottom_y, cell, highlight=(0, 0, 1, 1), dashed=True)

    arrow_color = "#A7A7A7"
    ax.add_patch(
        FancyArrow(
            6.5,
            5.1,
            2.2,
            0.0,
            width=0.10,
            head_width=0.35,
            head_length=0.35,
            length_includes_head=True,
            color=arrow_color,
        )
    )
    ax.add_patch(
        FancyArrow(
            6.5,
            2.3,
            2.2,
            0.0,
            width=0.10,
            head_width=0.35,
            head_length=0.35,
            length_includes_head=True,
            color=arrow_color,
        )
    )

    ax.text(7.6, 5.45, "最大池化", ha="center", va="center", fontsize=18)
    ax.text(7.6, 2.65, "平均池化", ha="center", va="center", fontsize=18)

    ax.text(2.95, 0.45, "输入数据", ha="center", va="center", fontsize=18)
    ax.text(11.1, 0.45, "输出数据", ha="center", va="center", fontsize=18)

    ax.text(7.45, 1.35, r"$\frac{3+2+1+4}{4}=2.5$", ha="center", va="center", fontsize=18)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=args.dpi)
    plt.close(fig)
    print(f"Saved figure to: {args.output.resolve()}")


if __name__ == "__main__":
    main()
