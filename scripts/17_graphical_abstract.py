"""Render an original vector schematic of the study-residual calibration workflow."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle
from plotting_support.scientific_palettes import get_palette


FOLDER = Path(__file__).resolve().parents[1] / "figures" / "Graphical_Abstract"
INK = "#26343c"
BIOCRUDE, CHAR, _, PALE_BLUE, LIGHT_BLUE, _, LIGHT_ROSE, SAGE = get_palette("au_nimno_soft_8")


def box(ax, x, y, w, h, fill, radius=0.28, edge=INK, lw=0.7):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={radius}",
                               facecolor=fill, edgecolor=edge, linewidth=lw))


def text(ax, x, y, value, size=9.3):
    ax.text(x, y, value, color=INK, fontsize=size, ha="center", va="center", linespacing=1.2)


def arrow(ax, start, end):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=8.0,
                                 linewidth=0.88, color=INK, shrinkA=0, shrinkB=0))


def line(ax, x, y, color=INK, width=0.75):
    ax.plot(x, y, color=color, lw=width, solid_capstyle="round")


def source_studies(ax):
    box(ax, 0.42, 1.10, 4.25, 7.8, LIGHT_BLUE, edge="none")
    for x, y in [(1.15, 5.34), (1.48, 4.98), (1.81, 4.62)]:
        ax.add_patch(Polygon([(x,y), (x+1.8,y), (x+1.8,y+1.86), (x+1.45,y+2.22), (x,y+2.22)],
                             facecolor="white", edgecolor=INK, linewidth=0.75))
        line(ax, [x+1.45, x+1.45, x+1.8], [y+2.22, y+1.86, y+1.86])
        for step in [1.52, 1.15, 0.78]:
            line(ax, [x+0.3, x+1.38], [y+step]*2, BIOCRUDE, width=1.05)
    ax.add_patch(Ellipse((2.46, 4.02), 1.35, 0.60, angle=30, color=SAGE))
    line(ax, [1.92, 2.95], [3.73, 4.27])
    text(ax, 2.54, 2.35, "Source studies")


def learn_from_sources(ax):
    box(ax, 6.02, 5.32, 5.92, 3.58, LIGHT_BLUE, edge="none")
    box(ax, 6.02, 1.10, 5.92, 3.58, LIGHT_ROSE, edge="none")
    for shift in [-1.25, 0.0, 1.25]:
        x = 8.98 + shift
        line(ax, [x-0.40, x, x+0.40], [7.04, 7.68, 7.04])
        for xx, yy in [(x-0.40, 7.04), (x, 7.68), (x+0.40, 7.04)]:
            ax.add_patch(Circle((xx, yy), 0.16, facecolor=BIOCRUDE, edgecolor=INK, linewidth=0.65))
    for size, angle, offset in [(1.95, 25, 0.0), (1.46, 25, 0.18), (0.88, 25, 0.34)]:
        ax.add_patch(Ellipse((8.83+offset, 3.12+0.12*offset), size, size/1.8, angle=angle, fill=False,
                             edgecolor=CHAR, linewidth=0.85))
    text(ax, 8.98, 6.19, "Pooled prediction")
    text(ax, 8.98, 1.97, "Residual covariance")
    arrow(ax, (4.69, 6.99), (6.00, 6.99))
    arrow(ax, (4.69, 2.89), (6.00, 2.89))


def calibrate_and_predict(ax):
    box(ax, 13.34, 4.13, 5.45, 3.55, PALE_BLUE, edge="none")
    box(ax, 13.34, 1.10, 5.45, 2.02, LIGHT_ROSE, edge="none")
    text(ax, 16.06, 7.04, "Candidate runs", size=8.7)
    for i, x in enumerate([14.44, 15.25, 16.06, 16.87, 17.68]):
        box(ax, x-0.21, 5.72, 0.42, 0.95, "white", radius=0.1)
        if i in (0, 2, 4):
            ax.add_patch(Rectangle((x-0.195, 5.83), 0.39, 0.36, facecolor=BIOCRUDE, edgecolor="none"))
        line(ax, [x-0.24, x+0.24], [6.67]*2, width=1.35)
    text(ax, 16.06, 4.97, "Measure selected runs", size=7.9)
    text(ax, 16.06, 2.11, "Local correction")
    arrow(ax, (16.06, 4.11), (16.06, 3.15))
    arrow(ax, (11.96, 2.11), (13.32, 2.11))
    line(ax, [11.96, 12.60, 12.60], [3.61, 3.61, 5.07])
    arrow(ax, (12.60, 5.07), (13.32, 5.07))
    box(ax, 20.48, 1.10, 4.13, 7.8, SAGE, edge="none")
    line(ax, [11.96, 12.64, 12.64, 19.54, 19.54], [7.57, 7.57, 8.47, 8.47, 6.40])
    arrow(ax, (19.54, 6.40), (20.46, 6.40))
    arrow(ax, (18.82, 2.11), (20.46, 2.11))
    box(ax, 21.19, 4.16, 0.95, 2.03, "white", radius=0.16)
    box(ax, 21.15, 6.14, 1.03, 0.27, PALE_BLUE, radius=0.08)
    box(ax, 21.33, 4.36, 0.67, 1.05, BIOCRUDE, radius=0.09, edge="none")
    ax.add_patch(Polygon([(22.67,4.25), (22.78,4.77), (23.07,4.84), (23.27,5.18),
                          (23.54,4.99), (23.72,5.34), (23.92,4.98), (24.17,4.72), (24.21,4.25)],
                         facecolor=CHAR, edgecolor=INK, linewidth=0.8))
    text(ax, 22.55, 2.89, "New-study\npredictions")


def build():
    FOLDER.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "Arial", "font.weight": "normal", "svg.fonttype": "none", "pdf.fonttype": 42})
    fig = plt.figure(figsize=(145/25.4, 52/25.4), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1], xlim=(0, 25), ylim=(.52, 9.48), aspect="equal")
    ax.axis("off")
    source_studies(ax)
    learn_from_sources(ax)
    calibrate_and_predict(ax)
    for suffix in ("png", "pdf", "svg"):
        meta={"Author":"CHONG LIU","Creator":"CHONG LIU"} if suffix=="pdf" else {"Creator":"CHONG LIU"}
        fig.savefig(FOLDER / f"Graphical_Abstract.{suffix}", dpi=600, facecolor="white", metadata=meta)
    sub=FOLDER/"subfigures"; sub.mkdir(exist_ok=True)
    for suffix in ("png","pdf"):
        fig.savefig(sub/f"Graphical_Abstract_a.{suffix}",dpi=600,facecolor="white",
                    metadata={"Author":"CHONG LIU","Creator":"CHONG LIU"})
    (sub/"README.md").write_text("The graphical abstract is one inseparable workflow. Its single panel is rendered directly from the vector source, not cropped.\n",encoding="utf-8")
    plt.close(fig)


if __name__ == "__main__":
    build()
