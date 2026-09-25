"""3D -> 2D projection helpers (MIP, thin-slab, and similar representations)."""

from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import zoom


PROJECTION_AXES = {"axial": 2, "coronal": 1, "sagittal": 0}


def maximum_intensity_projection(data: np.ndarray, axis: int | str) -> np.ndarray:
    """Collapse a 3D volume to 2D by taking the max value along one axis.

    axis may be an integer array axis or one of "axial"/"coronal"/"sagittal"
    (looked up in PROJECTION_AXES).
    """
    if isinstance(axis, str):
        axis = PROJECTION_AXES[axis]
    return np.max(data, axis=axis)


def resize_2d(img: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    """Resize 2D image via bilinear interpolation. No-op if already correct."""
    if img.shape == target_shape:
        return img
    factors = (target_shape[0] / img.shape[0], target_shape[1] / img.shape[1])
    return zoom(img, factors, order=1).astype(np.float32)


def load_volume(nifti_path: str | Path) -> np.ndarray:
    """Load a NIfTI volume as float32 array."""
    return nib.load(str(nifti_path)).get_fdata().astype(np.float32)


def top_z_crop(data: np.ndarray, top_pct: int) -> np.ndarray:
    """Keep the top N% of Z slices (axis 2), dropping the bottom.

    Assumes the region of interest sits at the top of the Z axis (e.g. a
    head scan with Z increasing from chest to skull) — anatomy/acquisition
    -specific, not a generic crop. Flip the volume first if your data's ROI
    is at the bottom instead.
    """
    z_total = data.shape[2]
    z_start = int(z_total * (1.0 - top_pct / 100.0))
    return data[:, :, z_start:]
