"""Sequential ID assignment, pluggable by naming scheme.

Generalizes the common pattern of a study with sequentially-numbered exam
folders (e.g. "EX0001", "case_042"): parse each folder name to a number,
assign a canonical ID, flag duplicates, and optionally synthesize rows for
missing numbers so gaps in the sequence are visible in the report.
"""
from typing import Callable

import pandas as pd


def assign_sequential_ids(
    df: pd.DataFrame,
    parse_id: Callable[[str], int | None],
    format_id: Callable[[int], str],
    *,
    name_column: str = "original_name",
    id_column: str = "id",
    status_column: str = "id_parse_status",
    fill_gaps: bool = True,
) -> pd.DataFrame:
    """
    Add id_column and status_column to df, assigning a canonical ID to each
    row from name_column. Optionally appends synthetic gap rows for every
    numeric ID absent from the parsed sequence.

    parse_id(name) -> numeric ID, or None if unparseable.
    format_id(numeric_id) -> canonical ID string (e.g. 42 -> "EX0042").

    Duplicates (same numeric ID) get alphabetical suffixes (A, B, C, ...) on
    the canonical ID, assigned in the order rows appear in df.

    status_column values:
      parsed        — unique, clean parse
      parsed_dedup  — duplicate numeric ID, suffix assigned
      unparseable   — parse_id returned None
      gap           — numeric ID absent from disk; all other columns are null
    """
    df = df.copy()

    # Plain Python list, not a pandas Series: a Series mixing ints and None
    # silently upcasts to float64 (None -> NaN), corrupting the IDs before
    # format_id ever sees them.
    numbers = [parse_id(name) for name in df[name_column]]

    counts: dict[int, int] = {}
    for n in numbers:
        if n is not None:
            counts[n] = counts.get(n, 0) + 1
    dup_numbers = {n for n, c in counts.items() if c > 1}

    ids = []
    statuses = []
    suffix_counters: dict[int, int] = {}

    for n in numbers:
        if n is None:
            ids.append(None)
            statuses.append("unparseable")
        elif n in dup_numbers:
            k = suffix_counters.get(n, 0)
            ids.append(f"{format_id(n)}{chr(ord('A') + k)}")
            statuses.append("parsed_dedup")
            suffix_counters[n] = k + 1
        else:
            ids.append(format_id(n))
            statuses.append("parsed")

    df[id_column] = ids
    df[status_column] = statuses

    if fill_gaps:
        present = {n for n in numbers if n is not None}
        if present:
            gap_numbers = sorted(set(range(min(present), max(present) + 1)) - present)
            if gap_numbers:
                gap_rows = pd.DataFrame({col: [None] * len(gap_numbers) for col in df.columns})
                gap_rows[id_column] = [format_id(n) for n in gap_numbers]
                gap_rows[status_column] = "gap"
                df = pd.concat([df, gap_rows], ignore_index=True)

    return df
