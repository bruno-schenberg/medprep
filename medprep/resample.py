"""Resampling and pad/crop to a fixed output shape — dataset variant generation.

Strategy: resample to a fixed isotropic (or anisotropic) spacing, then
pad/crop to a target voxel shape.
"""
import gc
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Spatial transform helpers (all operate on plain numpy arrays + 4x4 affine)
# ---------------------------------------------------------------------------

def get_spacing(affine: np.ndarray) -> np.ndarray:
    """Return voxel spacing (mm) as a 3-element array from a NIfTI affine."""
    return np.sqrt((affine[:3, :3] ** 2).sum(axis=0))


def scale_affine(affine: np.ndarray, old_spacing: np.ndarray, new_spacing: np.ndarray) -> np.ndarray:
    """Return a new affine with column vectors rescaled to new_spacing."""
    new_aff = affine.copy()
    for i in range(3):
        if old_spacing[i] > 0:
            new_aff[:3, i] *= new_spacing[i] / old_spacing[i]
    return new_aff


def resample_volume(
    data: np.ndarray,
    affine: np.ndarray,
    target_spacing: tuple[float, float, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Resample data to target_spacing using trilinear (order=1) zoom."""
    from scipy.ndimage import zoom as ndimage_zoom

    current_spacing = get_spacing(affine)
    zoom_factors = tuple(float(current_spacing[i]) / target_spacing[i] for i in range(3))
    resampled = ndimage_zoom(data, zoom_factors, order=1, prefilter=False)
    new_spacing = np.array(target_spacing, dtype=float)
    new_aff = scale_affine(affine, current_spacing, new_spacing)
    return resampled, new_aff


def pad_or_crop(
    data: np.ndarray,
    affine: np.ndarray,
    target_shape: tuple[int, int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Symmetrically pad (zeros) or crop each axis to reach target_shape exactly.

    Padding: adds zeros at both ends of the axis; affine origin shifts so that
    the padded array has a consistent world-space interpretation.
    Cropping: takes a centred window of target length.
    """
    result = data
    new_aff = affine.copy()

    for axis in range(3):
        curr = result.shape[axis]
        tgt = target_shape[axis]
        col = new_aff[:3, axis]  # direction vector for this axis

        if curr > tgt:
            start = (curr - tgt) // 2
            idx = [slice(None)] * 3
            idx[axis] = slice(start, start + tgt)
            result = result[tuple(idx)]
            new_aff[:3, 3] += start * col

        elif curr < tgt:
            pad_before = (tgt - curr) // 2
            pad_after = (tgt - curr) - pad_before
            pw = [(0, 0)] * 3
            pw[axis] = (pad_before, pad_after)
            result = np.pad(result, pw, mode="constant")
            new_aff[:3, 3] -= pad_before * col

    return result, new_aff


# ---------------------------------------------------------------------------
# Per-file worker (runs in subprocess)
# ---------------------------------------------------------------------------

def _process_one(
    src_path: str,
    out_path: str,
    output_shape: tuple[int, int, int],
    target_spacing_mm: tuple[float, float, float] | None,
) -> dict[str, Any]:
    """Process one NIfTI file. Returns a stats dict. Runs in a subprocess."""
    if Path(out_path).exists():
        return {"resample_status": "skipped_already_exists"}

    result: dict[str, Any] = {}

    try:
        img = nib.load(src_path)
        data = img.get_fdata(dtype=np.float32)
        affine = img.affine.copy()

        result["input_shape"] = "x".join(str(s) for s in data.shape)

        data, affine = resample_volume(data, affine, target_spacing_mm)
        pre_pad_shape = data.shape
        data, affine = pad_or_crop(data, affine, output_shape)

        # Safety: zoom rounding can produce off-by-one shape — correct here.
        if data.shape != tuple(output_shape):
            data, affine = pad_or_crop(data, affine, output_shape)

        if data.shape != tuple(output_shape):
            raise ValueError(f"Shape {data.shape} != target {output_shape}")

        # Pad/crop stats (fraction of voxels added/removed in the pad_or_crop step)
        pre_vol = int(np.prod(pre_pad_shape))
        post_vol = int(np.prod(output_shape))
        pct_pad = max(0.0, (post_vol - pre_vol) / post_vol) * 100 if post_vol > 0 else 0.0
        pct_crop = max(0.0, (pre_vol - post_vol) / pre_vol) * 100 if pre_vol > 0 else 0.0

        # Save
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        nib.save(nib.Nifti1Image(data, affine), out_path)

        final_spacing = get_spacing(affine)
        # Each worker process handles many files across its lifetime; drop
        # the (potentially large) volume array immediately rather than
        # waiting for the next allocation to trigger collection.
        del data
        gc.collect()

        result.update({
            "resample_status": "resampled",
            "actual_spacing_x": round(float(final_spacing[0]), 4),
            "actual_spacing_y": round(float(final_spacing[1]), 4),
            "actual_spacing_z": round(float(final_spacing[2]), 4),
            "pct_pad": round(pct_pad, 2),
            "pct_crop": round(pct_crop, 2),
        })

    except Exception as e:
        result["resample_status"] = "failed"
        result["error"] = str(e)

    return result


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def _collect_nifti_files(nifti_root: Path, sample_ids: set[str] | None) -> list[tuple[Path, str]]:
    """Return list of (path, class_str) for NIfTI files under nifti_root.

    nifti_root/<class>/<exam_id>.nii.gz

    If sample_ids is given, restricts to those exam IDs.
    """
    files = []
    for class_dir in sorted(nifti_root.iterdir()):
        if not class_dir.is_dir():
            continue
        for f in sorted(class_dir.glob("*.nii.gz")):
            exam_id = f.stem.replace(".nii", "")
            if sample_ids is not None and exam_id not in sample_ids:
                continue
            files.append((f, class_dir.name))
    return files


def run_resampling(
    nifti_root: Path,
    datasets_root: Path,
    variants: list[dict],
    sample_ids: set[str] | None = None,
    max_workers: int = 4,
) -> pd.DataFrame:
    """Generate all requested dataset variants from NIfTI files.

    Returns a DataFrame with one row per (exam, variant).

    nifti_root:    nifti/<class>/<exam_id>.nii.gz
    datasets_root: output root; each variant gets datasets/<name>/<class>/
    variants:      list of dicts with keys "name", "strategy", "output_shape",
                   and optionally "target_spacing_mm"
    sample_ids:    if given, restrict to these exam IDs
    max_workers:   parallel workers (one process per file)
    """
    files = _collect_nifti_files(nifti_root, sample_ids)
    if not files:
        logger.warning("No NIfTI files found.")
        return pd.DataFrame()

    rows = []

    for variant in variants:
        name = variant["name"]
        strategy = variant["strategy"]
        out_shape = tuple(variant["output_shape"])
        t_spacing = variant.get("target_spacing_mm")

        logger.info("[%s] strategy=%s  shape=%s  n=%d", name, strategy, out_shape, len(files))

        tasks = []
        for src_path, cls in files:
            exam_id = src_path.stem.replace(".nii", "")
            out_path = str(datasets_root / name / cls / src_path.name)
            tasks.append((exam_id, cls, str(src_path), out_path))

        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(
                    _process_one,
                    task[2],  # src_path
                    task[3],  # out_path
                    out_shape,
                    t_spacing,
                ): task
                for task in tasks
            }

            for future in as_completed(futures):
                exam_id, cls, src, out = futures[future]
                try:
                    stats = future.result()
                except Exception as e:
                    stats = {"resample_status": "failed", "error": str(e)}

                row = {
                    "exam_id": exam_id,
                    "class": cls,
                    "variant": name,
                    "strategy": strategy,
                    "output_shape": "x".join(str(s) for s in out_shape),
                }
                row.update(stats)
                rows.append(row)

                status = stats.get("resample_status", "unknown")
                if status == "failed":
                    logger.warning("FAILED %s: %s", exam_id, stats.get("error", ""))

    return pd.DataFrame(rows)
