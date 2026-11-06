"""Figure helpers: skill style + standalone-subfigure rendering (panels redrawn on own canvas)."""
from __future__ import annotations
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SUPPORT = Path(__file__).resolve().parent / 'plotting_support'
sys.path.insert(0, str(SUPPORT))
from ml_publication_style import configure_publication_style, style_axis, numeric_ticks  # noqa: E402
from scientific_palettes import get_palette, STYLE_COLORS  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
configure_publication_style()
PAL = get_palette("au_nimno_soft_8")
INK = STYLE_COLORS["ink"]
# fixed semantic roles shared by all figures
ROLE = {"random": PAL[1], "group": PAL[0], "P": PAL[3], "RE": PAL[2], "SC": PAL[1], "SC+RE": PAL[0],
        "naive": PAL[5], "FE": PAL[0], "OLS": PAL[2]}
EDGE = {PAL[3]: PAL[0], PAL[4]: PAL[0], PAL[5]: PAL[1], PAL[6]: PAL[1], PAL[7]: PAL[2]}  # pale fill -> palette outline


def edge(c):
    return EDGE.get(c, INK)


def save_combined_and_standalone(stem: str, panels: list, grid: tuple, figsize: tuple, single_size=(5.2, 4.4),
                                 gridspec_kw=None):
    """panels: list of callables f(ax). Saves combined + standalone subfigures (PNG+PDF)."""
    folder = FIG / stem
    sub = folder / "subfigures"
    sub.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(*grid, figsize=figsize, gridspec_kw=gridspec_kw or {})
    axes = axes.ravel() if hasattr(axes, "ravel") else [axes]
    for f, ax in zip(panels, axes):
        f(ax)
    for ax in axes[len(panels):]:
        ax.set_visible(False)
    fig.tight_layout()
    fig.savefig(folder / f"{stem}.png", dpi=600, facecolor="white", bbox_inches="tight")
    fig.savefig(folder / f"{stem}.pdf", facecolor="white", bbox_inches="tight")
    plt.close(fig)
    lines = ["# Subfigures", "", "Each panel is re-rendered on its own standalone canvas (not cropped from the combined figure).", ""]
    for i, f in enumerate(panels):
        s = "abcdefghijklmnopqrstuvwxyz"[i]
        fg, ax = plt.subplots(figsize=single_size)
        f(ax)
        fg.tight_layout()
        fg.savefig(sub / f"{stem}_{s}.png", dpi=600, facecolor="white", bbox_inches="tight")
        fg.savefig(sub / f"{stem}_{s}.pdf", facecolor="white", bbox_inches="tight")
        plt.close(fg)
        lines.append(f"- {s}: `{stem}_{s}.png`, `{stem}_{s}.pdf`")
    (sub / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return folder
