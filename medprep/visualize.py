"""Display-oriented loaders: mid-slice extraction, percentile normalization,
2D intensity centroid — shared by pipeline QC figures and one-off
justification/diagnostic scripts."""
from pathlib import Path

import nibabel as nib
import numpy as np
import pydicom

from .dicom_io import is_original


def percentile_normalize(arr: np.ndarray, low: float = 1, high: float = 99) -> np.ndarray:
    """Clip to the [low, high] percentile range of positive-valued pixels.

    No-op (returns arr unchanged) if there are no positive values.
    """
    pos = arr[arr > 0]
    if len(pos) == 0:
        return arr
    lo, hi = np.percentile(pos, [low, high])
    return np.clip(arr, lo, hi)


def load_mid_slice_nifti(
    path: str | Path,
) -> tuple[np.ndarray, tuple[float, float, float], tuple[int, int, int]]:
    """Return (percentile-clipped middle axial slice, voxel spacing, volume shape)
    for a NIfTI file."""
    img = nib.load(str(path))
    data = img.get_fdata()
    spacing = img.header.get_zooms()
    mid = data.shape[2] // 2
    slc = data[:, :, mid].T  # (Y, X) for imshow
    return percentile_normalize(slc), spacing, data.shape


def load_mid_slice_dicom(
    dcm_dir: str | Path,
    selected_series_uid: str | None = None,
) -> tuple[np.ndarray, dict]:
    """Return (percentile-clipped middle slice, metadata) for a DICOM series directory.

    Filters to selected_series_uid (if given) and excludes non-ORIGINAL
    (scout/derived) files. Returns a blank 64x64 placeholder slice and {}
    for metadata if no valid slices were found — a caller building a grid of
    thumbnails can render this as an empty tile without special-casing "no
    data" as a separate layout case.
    """
    dcm_dir = Path(dcm_dir)
    all_files = sorted(f for f in dcm_dir.iterdir() if f.is_file() and not f.name.startswith("."))
    if not all_files:
        return np.zeros((64, 64)), {}

    dcm_files = []
    for f in all_files:
        try:
            ds = pydicom.dcmread(str(f), stop_before_pixels=True)
        except Exception:
            continue
        if selected_series_uid is not None:
            try:
                if str(ds.SeriesInstanceUID) != selected_series_uid:
                    continue
            except AttributeError:
                continue
        if not is_original(ds):
            continue
        dcm_files.append(f)

    if not dcm_files:
        return np.zeros((64, 64)), {}

    ds0 = pydicom.dcmread(str(dcm_files[0]), stop_before_pixels=True)
    metadata = {
        "rows": int(getattr(ds0, "Rows", 0)),
        "columns": int(getattr(ds0, "Columns", 0)),
        "spacing_row": float(getattr(ds0, "PixelSpacing", [0, 0])[0]),
        "spacing_col": float(getattr(ds0, "PixelSpacing", [0, 0])[1]),
        "n_slices": len(dcm_files),
    }

    mid_idx = len(dcm_files) // 2
    ds = pydicom.dcmread(str(dcm_files[mid_idx]))
    slc = ds.pixel_array.astype(np.float32)

    return percentile_normalize(slc), metadata


def intensity_centroid_2d(slc: np.ndarray) -> tuple[float, float]:
    """Return (cx, cy) pixel centroid weighted by intensity.

    Thresholds at p20 of positive-valued pixels to ignore air/background,
    then computes the intensity-weighted centre of mass of the remainder.
    """
    pos = slc[slc > 0]
    if len(pos) == 0:
        return slc.shape[1] / 2, slc.shape[0] / 2
    thresh = np.percentile(pos, 20)
    mask = slc > thresh
    ys, xs = np.where(mask)
    weights = slc[mask]
    cx = float(np.average(xs, weights=weights))
    cy = float(np.average(ys, weights=weights))
    return cx, cy
