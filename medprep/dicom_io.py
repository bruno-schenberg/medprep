"""Shared DICOM I/O utilities."""

from pathlib import Path

import pydicom


def is_original(ds) -> bool:
    """True if ImageType is absent or contains 'ORIGINAL'."""
    try:
        return "ORIGINAL" in list(ds.ImageType)
    except AttributeError:
        return True


def load_headers(dcm_path: Path) -> tuple[list[pydicom.Dataset], int]:
    """Read all DICOM headers (no pixel data). Returns (valid_datasets, corrupt_count)."""
    datasets = []
    corrupt_count = 0
    for f in sorted(dcm_path.iterdir()):
        if not f.is_file():
            continue
        try:
            datasets.append(pydicom.dcmread(str(f), stop_before_pixels=True))
        except Exception:
            corrupt_count += 1
    return datasets, corrupt_count
