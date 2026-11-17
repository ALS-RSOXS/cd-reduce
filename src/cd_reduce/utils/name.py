"""Filename sequence inference utilities (compatible with pxr-reduce)."""

from __future__ import annotations

import re


def infer_index_regex(
    filenames: list[str],
    *,
    index_group: str = "index",
    prefix_group: str | None = None,
) -> str:
    """Infer a regular expression capturing the incrementing integer across FITS
    filenames.

    Parameters
    ----------
    filenames : list of str
        List of filenames (at least two entries).
    index_group : str, optional
        Name of the capture group for the numeric index. Default is ``"index"``.
    prefix_group : str or None, optional
        If supplied, capture the literal prefix before the index under this name.

    Returns
    -------
    str
        Anchored regular expression string.

    Raises
    ------
    ValueError
        If fewer than two filenames are given or no incrementing numeric block exists.
    """
    if len(filenames) < 2:
        raise ValueError("At least two filenames are required to infer a sequence.")

    sorted_names = sorted(filenames, key=_natural_sort_key)
    numeric_matches = [list(re.finditer(r"\d+", fn)) for fn in sorted_names]

    if any(not matches for matches in numeric_matches):
        raise ValueError("One or more filenames contain no digit blocks.")

    common_block_cnt = min(len(m) for m in numeric_matches)
    candidate_positions = []

    for pos in range(common_block_cnt):
        seq = [int(m[pos].group()) for m in numeric_matches]
        if all(seq[i + 1] - seq[i] == 1 for i in range(len(seq) - 1)):
            candidate_positions.append(pos)

    if not candidate_positions:
        raise ValueError("No integer block with a +1 progression found.")

    best_pos = max(
        candidate_positions,
        key=lambda p: (
            int(numeric_matches[-1][p].group()) - int(numeric_matches[0][p].group())
        ),
    )

    chosen_match = numeric_matches[0][best_pos]
    start, end = chosen_match.start(), chosen_match.end()

    prefix_lit = re.escape(sorted_names[0][:start])
    suffix_lit = re.escape(sorted_names[0][end:])

    if prefix_group and prefix_lit:
        prefix_pat = f"(?P<{prefix_group}>{prefix_lit})"
    else:
        prefix_pat = prefix_lit

    widths = [len(m[best_pos].group()) for m in numeric_matches]
    if len(set(widths)) == 1:
        width = widths[0]
        index_pat = rf"(?P<{index_group}>\d{{{width}}})"
    else:
        index_pat = rf"(?P<{index_group}>\d+)"

    return f"^{prefix_pat}{index_pat}{suffix_lit}$"


def _natural_sort_key(s: str) -> list[int | str]:
    """Generate a natural sort key ordering embedded integers numerically."""
    return [
        int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", s)
    ]
