import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Polygon, Rectangle


def para_points(x, y, w=1.8, h=0.52, skew=0.4):
    return [
        (x, y),
        (x + w, y),
        (x + w + skew, y + h),
        (x + skew, y + h),
    ]


def draw_para(
    ax,
    x,
    y,
    w=1.8,
    h=0.52,
    skew=0.4,
    edge="#1fb6ff",
    face="white",
    lw=2.2,
    text=None,
    text_dx=0.92,
    text_dy=0.26,
    fontsize=16,
):
    poly = Polygon(
        para_points(x, y, w, h, skew),
        closed=True,
        facecolor=face,
        edgecolor=edge,
        linewidth=lw,
        joinstyle="miter",
    )
    ax.add_patch(poly)
    if text:
        ax.text(
            x + text_dx,
            y + text_dy,
            text,
            fontsize=fontsize,
            fontstyle="italic",
            ha="center",
            va="center",
            family="serif",
        )
    return poly


def arrow(ax, start, end, color="black", lw=1.7, style="-|>", ms=10, ls="-", alpha=1.0):
    arr = FancyArrowPatch(
        start,
        end,
        arrowstyle=style,
        mutation_scale=ms,
        linewidth=lw,
        linestyle=ls,
        color=color,
        alpha=alpha,
        shrinkA=0,
        shrinkB=0,
    )
    ax.add_patch(arr)
    return arr


def dashed_curve(ax, pts, color, lw=2.6):
    xs, ys = zip(*pts)
    ax.plot(xs, ys, color=color, lw=lw, linestyle=(0, (5, 3)))


def draw_module_stub(ax, x, y, w=0.58, h=0.12, gap=0.1, angle=14, n=2):
    for i in range(n):
        rect = Rectangle((x, y + i * (h + gap)), w, h, angle=angle, facecolor="#bfbfbf", edgecolor="none", alpha=0.75)
        ax.add_patch(rect)


def main():
    out_dir = Path("runs/thesis_diagrams")
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(11.5, 5.8), dpi=200)
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 6.4)
    ax.axis("off")

    # panel borders
    ax.add_patch(Rectangle((0.1, 0.1), 6.3, 6.1, fill=False, edgecolor="#7f7f7f", linewidth=1.6, linestyle=(0, (2, 2))))
    ax.add_patch(Rectangle((6.4, 0.1), 6.5, 6.1, fill=False, edgecolor="#7f7f7f", linewidth=1.6, linestyle=(0, (2, 2))))
    ax.text(0.35, 5.75, "(a)", fontsize=20, family="serif")
    ax.text(6.75, 5.75, "(b)", fontsize=20, family="serif")

    # left panel feature pyramid
    left_x = 0.55
    left_levels = [0.7, 1.9, 3.1, 4.3, 5.25]
    left_w = [2.2, 2.0, 1.9, 1.75, 1.65]
    for y, w in zip(left_levels, left_w):
        draw_para(ax, left_x, y, w=w, edge="#22b8ff", lw=2.2)

    # bottom image placeholder
    img_poly = Polygon(
        [(0.55, 0.2), (2.85, 0.2), (3.22, 0.72), (0.92, 0.72)],
        closed=True,
        facecolor="#8ecae6",
        edgecolor="none",
        alpha=0.9,
    )
    ax.add_patch(img_poly)
    ax.text(1.9, 0.46, "Input", fontsize=12, color="white", ha="center", va="center", weight="bold")

    # vertical arrows left
    xmid = 1.95
    for y0, y1 in zip([0.72, 1.22, 2.42, 3.62, 4.82], [0.95, 2.15, 3.35, 4.55, 5.48]):
        arrow(ax, (xmid, y0), (xmid, y1), lw=1.5, ms=9)

    # middle p layers
    p_x = 4.1
    p_pos = {"P2": 0.68, "P3": 1.9, "P4": 3.1, "P5": 4.3}
    for name, y in p_pos.items():
        draw_para(ax, p_x, y, w=1.55, edge="#22b8ff", lw=2.2, text=name, text_dx=1.0, text_dy=0.27, fontsize=17)

    # horizontal links left->P
    src_y = [1.0, 2.2, 3.4, 4.6]
    dst_y = [0.96, 2.16, 3.36, 4.56]
    for sy, dy in zip(src_y, dst_y):
        ax.plot([2.95, 4.08], [sy, dy], color="black", lw=1.6)

    # top-down vertical links among P layers
    arrow(ax, (5.12, 4.3), (5.12, 3.62), lw=1.4, ms=9)
    arrow(ax, (5.12, 3.1), (5.12, 2.42), lw=1.4, ms=9)
    arrow(ax, (5.12, 1.9), (5.12, 1.22), lw=1.4, ms=9)

    # dashed emphasis paths
    dashed_curve(ax, [(3.95, 4.08), (3.2, 3.7), (2.65, 3.0), (2.45, 1.25)], color="#ff3b30")
    dashed_curve(ax, [(2.55, 1.15), (3.7, 1.15), (5.55, 1.48), (7.15, 3.0)], color="#34c759")
    ax.text(5.0, 4.76, "✦", color="#ff3b30", fontsize=14, ha="center", va="center")

    # right panel neck outputs
    right_x = 7.05
    n_pos = {"N2": 0.9, "N3": 2.05, "N4": 3.25, "N5": 4.45}
    for name, y in n_pos.items():
        draw_para(ax, right_x, y, w=1.95, edge="#ff8c2a", lw=2.2, text=name, text_dx=0.95, text_dy=0.3, fontsize=17)
        draw_module_stub(ax, right_x + 1.45, y + 0.1)

    # links from P to N
    ax.plot([5.65, 7.03], [0.95, 0.95], color="black", lw=1.6)
    ax.plot([5.65, 7.03], [4.55, 4.55], color="black", lw=1.6)
    ax.plot([5.65, 7.03], [3.35, 3.35], color="black", lw=1.6)

    # vertical and horizontal aggregation in N
    arrow(ax, (8.7, 1.42), (8.7, 2.1), lw=1.4, ms=9)
    arrow(ax, (8.7, 2.58), (8.7, 3.28), lw=1.4, ms=9)
    arrow(ax, (8.7, 3.78), (8.7, 4.48), lw=1.4, ms=9)
    arrow(ax, (8.15, 3.52), (8.15, 4.25), color="#34c759", lw=2.0, ms=10, ls=(0, (5, 3)))

    # extra right-going lines
    for y in [1.18, 2.33, 3.53, 4.73]:
        ax.plot([9.42, 10.9], [y, y], color="black", lw=1.5)

    fig.tight_layout(pad=0.2)
    png = out_dir / "neck_diagram_style.png"
    svg = out_dir / "neck_diagram_style.svg"
    fig.savefig(png, bbox_inches="tight", facecolor="white")
    fig.savefig(svg, bbox_inches="tight", facecolor="white")
    print(png)
    print(svg)


if __name__ == "__main__":
    main()
