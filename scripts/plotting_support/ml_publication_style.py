"""
Shared publication figure helpers for Nature-style ML figures.

Use these helpers in every generated figure script unless the journal requires
a different style. They enforce visible tick marks, bold readable tick labels,
high-resolution export, and per-panel subfigure export.
"""

from __future__ import annotations

import string
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from scientific_palettes import PALETTES, STYLE_COLORS, get_palette


def configure_publication_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "figure.dpi": 160,
            "savefig.dpi": 600,
            "axes.labelsize": 15.5,
            "axes.labelweight": "bold",
            "xtick.labelsize": 12.5,
            "ytick.labelsize": 12.5,
            "legend.fontsize": 11.5,
            "axes.linewidth": 1.7,
            "xtick.major.size": 7.2,
            "ytick.major.size": 7.2,
            "xtick.major.width": 1.55,
            "ytick.major.width": 1.55,
            "xtick.minor.size": 4.2,
            "ytick.minor.size": 4.2,
            "xtick.minor.width": 1.15,
            "ytick.minor.width": 1.15,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def style_axis(ax, xlabel: str | None = None, ylabel: str | None = None,
               tick_size: float = 12.5, label_size: float = 15.5) -> None:
    ax.grid(False)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["bottom", "left"]:
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_color(STYLE_COLORS["ink"])
        ax.spines[spine].set_linewidth(1.7)
    ax.xaxis.set_ticks_position("bottom")
    ax.yaxis.set_ticks_position("left")
    ax.tick_params(axis="x", which="major", bottom=True, top=False, colors=STYLE_COLORS["ink"],
                   width=1.55, length=7.2, direction="out", labelsize=tick_size)
    ax.tick_params(axis="y", which="major", left=True, right=False, colors=STYLE_COLORS["ink"],
                   width=1.55, length=7.2, direction="out", labelsize=tick_size)
    ax.tick_params(axis="both", which="minor", colors=STYLE_COLORS["ink"], width=1.15,
                   length=4.2, direction="out")
    for tick in ax.xaxis.get_major_ticks():
        tick.tick1line.set_visible(True)
        tick.tick1line.set_markersize(7.2)
        tick.tick1line.set_markeredgewidth(1.55)
    for tick in ax.yaxis.get_major_ticks():
        tick.tick1line.set_visible(True)
        tick.tick1line.set_markersize(7.2)
        tick.tick1line.set_markeredgewidth(1.55)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")
        label.set_fontsize(tick_size)
    ax.xaxis.label.set_fontweight("bold")
    ax.yaxis.label.set_fontweight("bold")
    ax.xaxis.label.set_size(label_size)
    ax.yaxis.label.set_size(label_size)
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontweight="bold", fontsize=label_size, labelpad=7)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontweight="bold", fontsize=label_size, labelpad=7)


def numeric_ticks(ax, x: bool = True, y: bool = True, n: int = 5) -> None:
    """Keep numeric axes readable; use 3-6 major ticks unless the domain differs."""
    if x:
        ax.xaxis.set_major_locator(MaxNLocator(nbins=n, prune=None))
    if y:
        ax.yaxis.set_major_locator(MaxNLocator(nbins=n, prune=None))


def _panel_letters() -> list[str]:
    letters = list(string.ascii_lowercase)
    return letters + [a + b for a in letters for b in letters]


def export_subfigures(fig: plt.Figure, folder: str | Path, stem: str,
                      dpi: int = 600, pad: float = 1.06) -> Path:
    """Save each visible subplot into `folder/subfigures`.

    The crop comes from the final combined figure, so labels, ticks, legends,
    and colorbars match the exported main figure. Colorbar-only axes are skipped.
    """
    folder = Path(folder)
    subdir = folder / "subfigures"
    subdir.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    labels = _panel_letters()
    exported = []
    for ax in fig.axes:
        if not ax.get_visible() or str(ax.get_label()).startswith("<colorbar"):
            continue
        bbox = ax.get_tightbbox(renderer)
        if bbox is None:
            continue
        bbox = bbox.expanded(pad, pad).transformed(fig.dpi_scale_trans.inverted())
        suffix = labels[len(exported)]
        png = subdir / f"{stem}_{suffix}.png"
        pdf = subdir / f"{stem}_{suffix}.pdf"
        fig.savefig(png, dpi=dpi, bbox_inches=bbox, facecolor="white")
        fig.savefig(pdf, bbox_inches=bbox, facecolor="white")
        exported.append((suffix, png.name, pdf.name))
    readme = subdir / "README.md"
    lines = ["# Subfigures", "", "Extracted from the final combined figure export.", ""]
    for suffix, png_name, pdf_name in exported:
        lines.append(f"- {suffix}: `{png_name}`, `{pdf_name}`")
    readme.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return subdir


def save_figure_with_subfigures(fig: plt.Figure, folder: str | Path, stem: str,
                                dpi: int = 600, close: bool = True,
                                export_panels: bool = True) -> None:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    fig.savefig(folder / f"{stem}.png", dpi=dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(folder / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    if export_panels:
        export_subfigures(fig, folder, stem, dpi=dpi)
    if close:
        plt.close(fig)
