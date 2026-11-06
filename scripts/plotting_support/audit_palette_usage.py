"""Audit declared data-series colours and artists against a selected palette.

No neutral-series exemption is provided. Structural ink should not be registered
as a data series. This helper does not establish audit completeness or inspect
saved PNG/PDF files; export inspection remains necessary.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from matplotlib.collections import Collection
from matplotlib.colors import to_hex, to_rgba
from matplotlib.container import BarContainer
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from scientific_palettes import get_palette


def _palette_hex(palette_name, role, color):
    rgba = to_rgba(color)
    for allowed in get_palette(palette_name):
        if np.allclose(rgba[:3], to_rgba(allowed)[:3], atol=1e-8, rtol=0):
            return allowed
    raise ValueError(
        f"{role}: {to_hex(rgba).upper()} is not in {palette_name}; "
        "baseline/control data series have no neutral-colour exemption"
    )


def validate_series_colors(palette_name, series_colors):
    """Return canonical HEX roles, rejecting extra hues even for baselines."""
    if not series_colors:
        raise ValueError("Declare at least one data-series colour")
    return {role: _palette_hex(palette_name, role, color)
            for role, color in series_colors.items()}


def _artists(value):
    if isinstance(value, BarContainer):
        yield from value.patches
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _artists(item)
    else:
        yield value


def _series_colours(artist):
    if isinstance(artist, Line2D):
        colours = []
        if artist.get_linestyle() not in (None, "None", "none", "", " "):
            colours.append(artist.get_color())
        if artist.get_marker() not in (None, "None", "none", "", " "):
            face = artist.get_markerfacecolor()
            if to_rgba(face)[3] == 0 or np.allclose(to_rgba(face)[:3], (1, 1, 1)):
                face = artist.get_markeredgecolor()
            colours.append(face)
        return colours
    if isinstance(artist, Patch):
        return [artist.get_facecolor()]
    if isinstance(artist, Collection):
        faces = artist.get_facecolors()
        visible = [c for c in faces if c[3] > 0]
        # Hollow scatter markers encode the series through their coloured edge.
        if not visible or all(np.allclose(c[:3], (1, 1, 1)) for c in visible):
            return list(artist.get_edgecolors())
        return visible
    raise TypeError(f"Unsupported data artist: {type(artist).__name__}")


def audit_series_artists(palette_name, series_artists, expected_colors=None):
    """Check declared data marks and legend proxies, including role agreement.

    Values may be a BarContainer, line, scatter/fill collection, patch, or nested
    lists/tuples of these. Filled-marker outlines are structural and not checked.
    Supply all series and their legend proxies; unsupported artists fail visibly.
    """
    if not series_artists:
        raise ValueError("No data-series artists declared")
    expected = (validate_series_colors(palette_name, expected_colors)
                if expected_colors is not None else None)
    if expected is not None and set(expected) != set(series_artists):
        raise ValueError("Expected role map and declared artist roles differ")
    checks = []
    for role, values in series_artists.items():
        count = 0
        for artist in _artists(values):
            colours = _series_colours(artist)
            if not colours:
                raise ValueError(f"{role}: artist has no inspectable series colour")
            actual = [_palette_hex(palette_name, role, c) for c in colours]
            if expected is not None and any(c != expected[role] for c in actual):
                raise ValueError(f"{role}: actual {actual} differs from {expected[role]}")
            checks.append({"role": role, "artist": type(artist).__name__,
                           "base_colours": sorted(set(actual))})
            count += 1
        if not count:
            raise ValueError(f"{role}: no artists supplied")
    return {"passed": True, "palette": palette_name, "checks": checks,
            "scope": "declared artists only; exported files not inspected"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--palette", required=True)
    parser.add_argument("--series-map", type=Path, required=True,
                        help="JSON object mapping data-series labels to colours")
    args = parser.parse_args()
    roles = json.loads(args.series_map.read_text(encoding="utf-8-sig"))
    result = validate_series_colors(args.palette, roles)
    print(json.dumps({"palette": args.palette, "series_colors": result,
                      "scope": "role map only; artists and exports not inspected"}, indent=2))


if __name__ == "__main__":
    main()
