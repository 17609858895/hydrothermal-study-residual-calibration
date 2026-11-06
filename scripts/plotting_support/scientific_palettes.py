"""Reusable scientific palettes extracted from the user's reference figures."""

from __future__ import annotations

import argparse
import re


PALETTES: dict[str, tuple[str, ...]] = {
    # Existing author-reference palettes.
    "nature3": ("#369E9D", "#D37DB4", "#F3B491", "#6666CC", "#CCB743"),
    "nature1": ("#6BB952", "#EC748B", "#C4A751", "#36ACA2", "#6EB1DE", "#B67FB3"),
    "nature5": ("#F6B7C6", "#A2DADE", "#FEAF8A", "#A2C2E2", "#D8CBF0"),
    "nature6": ("#FFD47F", "#F7C1CF", "#7B92C7", "#ADD9EE"),
    "nature4": ("#DF9E9B", "#99BADF", "#D8E7CA", "#99CDCE", "#999ACD", "#FFD0E9", "#E6AB02", "#A6761D"),
    "nature3_alt": ("#169063", "#F29538", "#5F93CE", "#C96687"),
    "diverging": ("#CE6A6C", "#2278B4"),
    "nature_new": ("#DF9E9B", "#99BADF", "#D8E7CA", "#99CDCE", "#999ACD", "#FFD0E9", "#F6D58B", "#C7A3CC"),
    "nature_new1": ("#0071C2", "#D75615", "#EDB11A", "#7E318A", "#78AB31", "#2A77AC", "#D55535"),

    # Exact RGB-labelled palettes from the attached figures.
    "au_nimno_soft_8": ("#79C1E4", "#E68282", "#B2D362", "#BAE1F3", "#D4EAF8", "#EECDD5", "#F8E6E4", "#D1E4A6"),
    "battery_pastel_10": ("#84C7EE", "#EFAFCD", "#E1D4E8", "#F0B688", "#A59AC6", "#B4DAC5", "#B9D3EE", "#8CBABF", "#A4CA8B", "#5670A5"),
    "pcie_vivid_9": ("#1C3885", "#4F8CBB", "#F4A25C", "#DD542F", "#93DCB0", "#12AF62", "#008280", "#C7BEDF", "#3F68DD"),
    "hydrogel_soft_7": ("#7CC4B8", "#EFB1AA", "#99A4BE", "#9FD6E1", "#F1C9B7", "#F1E1AA", "#E0B2C2"),
    "biofilm_pastel_9": ("#F7C2C3", "#BED4EA", "#A8D298", "#F5DFA9", "#D3B7D7", "#D8E5F1", "#CAE2C1", "#F9E9C9", "#E5D1E5"),
    "ba_eu_ir_o_soft_8": ("#73BAE3", "#F5CBA2", "#99CCAD", "#EBAAA3", "#FAE4CD", "#C4DEF1", "#E6984C", "#0B79BD"),
    "ptsn_warm_cool_10": ("#E17878", "#D8292C", "#F0BAB9", "#E08A2E", "#CB9A1A", "#65ACA8", "#7EB5DC", "#4E85AC", "#BEC3E1", "#7D89BD"),
    "polarization_soft_10": ("#F7E0CF", "#C6DEED", "#D1E4CF", "#D7D894", "#9FC0D6", "#58A8D7", "#E39C63", "#3F719D", "#5A8B3B", "#A9C37F"),
    "thermal_gradient_8": ("#55AFE2", "#A3E3FF", "#8BCFB5", "#B9DA9A", "#686789", "#F7A7A6", "#E96A6A", "#FE8300"),
    "multilevel_memory_10": ("#388AC2", "#96C145", "#D3CB41", "#98B8D7", "#9FC488", "#F2DB96", "#B55920", "#D69D2F", "#F2CACA", "#FAE3CE"),

    # JPG bar-fill palettes: median colors extracted from the bar interiors.
    "bar_soft_balanced_5": ("#E9D29C", "#D6E4CB", "#97B4D4", "#DDC6E0", "#9B72AA"),
    "bar_pastel_5": ("#B4AED4", "#AFE7FF", "#B8D1CD", "#EAAFA7", "#FEEFB8"),
    "bar_mauve_5": ("#568EA9", "#D9CDE3", "#C8B2C7", "#9F80BC", "#CDA0C7"),
    "bar_blue_green_5": ("#A4CDED", "#6996BF", "#B8C3E3", "#80BEA5", "#317A5D"),
}


# User-preferred palettes, in default selection order. ``soft_8`` is the user's
# shorthand for the canonical registry key ``au_nimno_soft_8``.
PREFERRED_PALETTES: tuple[str, ...] = ("au_nimno_soft_8", "nature_new")
DEFAULT_PALETTE = PREFERRED_PALETTES[0]


# Structural neutrals are for text, axes, outlines and statistical/reference marks,
# never automatic exceptions for baseline/control data-series colours.
STYLE_COLORS = {"ink": "#30343B", "muted": "#6F7782", "light": "#F3F5F7"}
# Legacy mixed-palette demo roles, retained for old template compatibility only.
# They are NOT a named palette and are not compliant with a user-selected palette.
# New/adapted figures must map every data series via get_palette() and validate it
# with audit_palette_usage.py; do not copy 'muted' into baseline/control series.
PLOT_COLORS = {
    "blue": "#7EB5DC",
    "green": "#65ACA8",
    "red": "#D8292C",
    "navy": "#4E85AC",
    "coral": "#E17878",
    "teal": "#008280",
    "lavender": "#7D89BD",
    "gold": "#CB9A1A",
    "orange": "#E08A2E",
    "purple": "#9F80BC",
    "brown": "#B55920",
    "sand": "#F2DB96",
    **STYLE_COLORS,
}


def get_palette(name: str, n: int | None = None, reverse: bool = False) -> list[str]:
    """Return a palette without silently repeating colors."""
    if name not in PALETTES:
        raise KeyError(f"Unknown palette {name!r}; choose from {sorted(PALETTES)}")
    colors = list(PALETTES[name])
    if reverse:
        colors.reverse()
    if n is None:
        return colors
    if n < 1:
        raise ValueError("n must be positive")
    if n > len(colors):
        raise ValueError(f"Palette {name!r} has {len(colors)} colors, fewer than requested n={n}")
    return colors[:n]


def validate_palettes() -> list[str]:
    """Return validation problems; an empty list means the registry is valid."""
    problems: list[str] = []
    pattern = re.compile(r"^#[0-9A-F]{6}$")
    for name, colors in PALETTES.items():
        if not colors:
            problems.append(f"{name}: empty palette")
        if len(colors) != len(set(colors)):
            problems.append(f"{name}: duplicate colors")
        for color in colors:
            if not pattern.fullmatch(color):
                problems.append(f"{name}: invalid HEX {color}")
    for name in PREFERRED_PALETTES:
        if name not in PALETTES:
            problems.append(f"preferred palette {name!r}: missing from registry")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="List palette names and sizes")
    parser.add_argument("--name", choices=sorted(PALETTES), help="Print one palette")
    parser.add_argument("--n", type=int, help="Return only the first n colors")
    parser.add_argument("--reverse", action="store_true")
    args = parser.parse_args()
    problems = validate_palettes()
    if problems:
        raise SystemExit("\n".join(problems))
    if args.name:
        print(" ".join(get_palette(args.name, args.n, args.reverse)))
    else:
        for name, colors in PALETTES.items():
            print(f"{name}: {len(colors)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
