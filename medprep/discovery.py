"""Archive extraction and DICOM exam-folder discovery.

Expected layout for discover_exams:
    staging_root/
      <archive_stem>/      <- one directory per extracted archive
        <exam_folder>/      <- candidate exam (one row in the output)
          *.dcm  (or subfolders with *.dcm)

If the real data has a different nesting depth, adjust or replace
discover_exams for that layout.
"""
import logging
import shutil
import zipfile
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def extract_archives(source_dir: Path, staging_root: Path) -> None:
    """Extract every .zip in source_dir into its own subdirectory under staging_root.

    Extraction is atomic (extract to a temp dir, then rename to dest): an
    interrupted run leaves no partial `dest` directory that a later run would
    mistake for "already extracted" and skip forever.
    """
    staging_root.mkdir(parents=True, exist_ok=True)

    for zip_path in sorted(source_dir.glob("*.zip")):
        dest = staging_root / zip_path.stem
        if dest.exists():
            logger.info("skipping %s (already extracted)", zip_path.name)
            continue

        logger.info("extracting %s ...", zip_path.name)
        tmp_dest = staging_root / f".{zip_path.stem}.tmp"
        if tmp_dest.exists():
            shutil.rmtree(tmp_dest)
        tmp_dest.mkdir()
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(tmp_dest)
        tmp_dest.rename(dest)


# ---------------------------------------------------------------------------
# DICOM detection
# ---------------------------------------------------------------------------

def _is_dicom(path: Path) -> bool:
    if path.suffix.lower() == ".dcm":
        return True
    if path.suffix == "":
        try:
            if path.stat().st_size < 132:
                return False
            with open(path, "rb") as f:
                f.seek(128)
                return f.read(4) == b"DICM"
        except OSError:
            return False
    return False


def _dicom_files_in(folder: Path) -> list[Path]:
    """Return all DICOM files directly inside folder (non-recursive)."""
    return [p for p in folder.iterdir() if p.is_file() and _is_dicom(p)]


# ---------------------------------------------------------------------------
# Folder classification
# ---------------------------------------------------------------------------

def _classify_exam_folder(folder: Path, archive_source: str) -> dict:
    """
    Classify a single candidate exam folder.

    A 'dicom_group' is any location (the folder itself, or a direct subfolder)
    that contains at least one DICOM file.

    dicom_groups == 0  ->  empty
    dicom_groups == 1  ->  ready
    dicom_groups  > 1  ->  review_duplicate
    """
    subfolders = sorted(p for p in folder.iterdir() if p.is_dir())

    # Each location that contains DICOMs counts as one group.
    dicom_groups: list[Path] = []
    if _dicom_files_in(folder):
        dicom_groups.append(folder)
    for sub in subfolders:
        if _dicom_files_in(sub):
            dicom_groups.append(sub)

    count = len(dicom_groups)
    if count == 0:
        status, dcm_path = "empty", None
    elif count == 1:
        status, dcm_path = "ready", dicom_groups[0]
    else:
        status, dcm_path = "review_duplicate", dicom_groups[0]

    return {
        "original_name": folder.name,
        "zip_source": archive_source,
        "dcm_path": str(dcm_path) if dcm_path else None,
        "folder_status": status,
        "subfolder_count": len(subfolders),
    }


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

_COLUMNS = ["original_name", "zip_source", "dcm_path", "folder_status", "subfolder_count"]


def discover_exams(staging_root: Path) -> pd.DataFrame:
    """Walk staging_root and return a DataFrame classifying every candidate exam folder."""
    rows = []

    for archive_dir in sorted(staging_root.iterdir()):
        if not archive_dir.is_dir() or archive_dir.name.startswith("."):
            continue

        for exam_folder in sorted(archive_dir.iterdir()):
            if not exam_folder.is_dir() or exam_folder.name == "__MACOSX":
                continue
            rows.append(_classify_exam_folder(exam_folder, archive_source=archive_dir.name))

    if not rows:
        return pd.DataFrame(columns=_COLUMNS)

    return pd.DataFrame(rows, columns=_COLUMNS)
