"""DICOM series -> NIfTI conversion, and centroid-based XY FOV cropping."""
from pathlib import Path

import numpy as np
import pydicom

from .dicom_io import is_original
from .visualize import intensity_centroid_2d


def resolve_dicom_file_list(
    dcm_path: Path, selected_series_uid: str | None
) -> tuple[list[Path], int]:
    """Return (sorted DICOM files to convert, unreadable_header_count) for one exam.

    Loads headers (no pixel data) to filter by series UID and exclude scouts
    (non-ORIGINAL ImageType). Files whose header can't be read are counted
    rather than silently dropped, mirroring dicom_io.load_headers's contract.
    """
    files = []
    unreadable_count = 0
    for f in sorted(dcm_path.iterdir()):
        if not f.is_file():
            continue
        try:
            ds = pydicom.dcmread(str(f), stop_before_pixels=True)
        except Exception:
            unreadable_count += 1
            continue

        if selected_series_uid is not None:
            try:
                if str(ds.SeriesInstanceUID) != selected_series_uid:
                    continue
            except AttributeError:
                continue

        if not is_original(ds):
            continue

        files.append(f)

    return files, unreadable_count


def load_dicom_series_as_array(series_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load a DICOM series directory as a (numpy_data, affine) pair via MONAI + ITK.

    ITK handles series reconstruction, affine matrix construction, and DICOM
    edge cases automatically. series_path must be a directory (ITKReader
    requirement for 3D series reconstruction).
    """
    import itk
    from monai.data import ITKReader
    from monai.transforms import LoadImage

    loader = LoadImage(reader=ITKReader(), image_only=True)
    itk.ProcessObject.SetGlobalWarningDisplay(False)
    try:
        image_data = loader(str(series_path))
    finally:
        itk.ProcessObject.SetGlobalWarningDisplay(True)
    numpy_data = image_data.numpy().astype(np.float32)
    affine = image_data.affine.numpy()
    return numpy_data, affine


# ---------------------------------------------------------------------------
# Centroid crop
# ---------------------------------------------------------------------------

def xy_intensity_centroid(volume: np.ndarray) -> tuple[float, float]:
    """Intensity-weighted XY centroid from the middle half of slices.

    Projects slices z25%-z75% as a mean to get a stable slab free from the
    weak signal at the top and bottom of the scan, then delegates the
    threshold-and-weighted-centroid math to visualize.intensity_centroid_2d
    (transposed to match its (row=Y, col=X) convention).

    volume: shape (X, Y, Z).
    Returns (cx, cy) in voxel coordinates along the X and Y axes.
    """
    z0 = volume.shape[2] // 4
    z1 = volume.shape[2] * 3 // 4
    slab = volume[:, :, z0:z1].mean(axis=2)  # (X, Y)
    return intensity_centroid_2d(slab.T)


def centroid_crop_xy(
    volume: np.ndarray,
    affine: np.ndarray,
    target_fov_mm: float,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Resize each XY axis to exactly target_fov_mm by cropping or padding.

    Each axis is handled independently:
    - Oversized axis: cropped to target_fov_mm centred on the intensity
      centroid. The window is clamped to the image boundary (worst case:
      slightly off-centre).
    - Undersized axis: zero-padded symmetrically to target_fov_mm.
    - Exact axis: unchanged.

    All outputs have the same XY physical size (target_fov_mm x target_fov_mm).

    The affine origin is updated to reflect the new (0, 0, 0) voxel position,
    keeping the output physically correct.

    Returns (volume, updated_affine, x0, y0).
    x0 and y0 are the crop start voxels (0 if that axis was padded or unchanged).
    """
    nx, ny, nz = volume.shape
    spacing_x = float(np.linalg.norm(affine[:3, 0]))
    spacing_y = float(np.linalg.norm(affine[:3, 1]))

    tvx = int(round(target_fov_mm / spacing_x))
    tvy = int(round(target_fov_mm / spacing_y))

    if nx == tvx and ny == tvy:
        return volume, affine, 0, 0

    # Centroid needed only if at least one axis must be cropped.
    if nx > tvx or ny > tvy:
        cx, cy = xy_intensity_centroid(volume)

    # --- X axis ---
    if nx > tvx:
        x0 = int(round(cx - tvx / 2))
        x0 = max(0, min(x0, nx - tvx))
        data = volume[x0:x0 + tvx, :, :]
        shift_x = x0
    elif nx < tvx:
        pad_before = (tvx - nx) // 2
        pad_after = (tvx - nx) - pad_before
        data = np.pad(volume, [(pad_before, pad_after), (0, 0), (0, 0)])
        x0, shift_x = 0, -pad_before
    else:
        data = volume
        x0, shift_x = 0, 0

    # --- Y axis ---
    if ny > tvy:
        y0 = int(round(cy - tvy / 2))
        y0 = max(0, min(y0, ny - tvy))
        data = data[:, y0:y0 + tvy, :]
        shift_y = y0
    elif ny < tvy:
        pad_before = (tvy - ny) // 2
        pad_after = (tvy - ny) - pad_before
        data = np.pad(data, [(0, 0), (pad_before, pad_after), (0, 0)])
        y0, shift_y = 0, -pad_before
    else:
        y0, shift_y = 0, 0

    shift = affine[:3, :3] @ np.array([shift_x, shift_y, 0], dtype=float)
    new_affine = affine.copy()
    new_affine[:3, 3] += shift

    return data, new_affine, x0, y0
