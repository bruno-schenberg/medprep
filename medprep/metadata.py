"""Join a project's metadata CSV (labels, demographics, scanner info, ...)
onto a pipeline DataFrame, by a configurable ID scheme."""
import logging
from pathlib import Path
from typing import Callable

import pandas as pd

logger = logging.getLogger(__name__)


def join_metadata(
    df: pd.DataFrame,
    metadata_csv: Path,
    *,
    id_column: str,
    metadata_id_column: str,
    normalize_metadata_id: Callable[[str], str | None],
    strip_id_suffix: Callable[[str], str] = lambda x: x,
    rename_columns: dict[str, str] | None = None,
    keep_columns: list[str] | None = None,
) -> pd.DataFrame:
    """
    Left-join metadata_csv onto df.

    df[id_column] values are stripped of any dedup suffix (via
    strip_id_suffix) before joining, so e.g. "EX0250A" and "EX0250" both
    match the "EX0250" row in the metadata CSV.

    id_column:             join key column in df (e.g. "exam_id")
    metadata_id_column:    join key column in metadata_csv (e.g. "exam")
    normalize_metadata_id: canonicalize a raw metadata_csv ID to match
                            df's ID scheme (e.g. "EX001" -> "EX0001"),
                            or None if unparseable (row is dropped)
    strip_id_suffix:       remove a dedup suffix from a df ID (e.g.
                            "EX0250A" -> "EX0250"); identity by default
    rename_columns:        renamed on the metadata_csv side before the join
                            (e.g. {"Age": "age"})
    keep_columns:          metadata_csv columns to keep (post-rename); None
                            keeps all columns

    Logs a warning listing metadata_csv entries with no matching row in df.
    """
    metadata_csv = Path(metadata_csv)
    metadata = pd.read_csv(metadata_csv)
    if rename_columns:
        metadata = metadata.rename(columns=rename_columns)

    metadata["_join_id"] = metadata[metadata_id_column].map(normalize_metadata_id)
    metadata = metadata.dropna(subset=["_join_id"])

    if keep_columns is not None:
        keep = ["_join_id"] + [c for c in keep_columns if c in metadata.columns]
        metadata = metadata[keep]

    df = df.copy()
    df["_join_id"] = df[id_column].map(
        lambda x: strip_id_suffix(x) if pd.notna(x) else None
    )

    merged = df.merge(metadata, on="_join_id", how="left").drop(columns=["_join_id"])

    disk_ids = set(df[id_column].dropna().map(strip_id_suffix))
    unmatched = metadata[~metadata["_join_id"].isin(disk_ids)]
    if not unmatched.empty:
        logger.warning(
            "%d %s entries have no matching row: %s",
            len(unmatched), metadata_csv.name, ", ".join(unmatched["_join_id"].tolist()),
        )

    return merged
