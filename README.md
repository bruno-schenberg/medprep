# medprep

Medical image preprocessing library. Handles the common steps between raw DICOM archives and ML-ready datasets.

## What it does

- **DICOM discovery** — scan directories for DICOM archives, extract exams, assign sequential IDs
- **QC checks** — scanner-agnostic series validation (spacing, extent, orientation, scout filtering)
- **DICOM → NIfTI conversion** — ITK-based conversion with optional centroid-based FOV cropping
- **Resampling** — resample to isotropic or target spacing, pad/crop to target shape
- **3D → 2D projections** — MIP (maximum intensity projection) and thin-slab generation
- **Visualization** — mid-slice loaders, percentile normalization, intensity centroid overlays
- **Metadata** — join external metadata CSVs with configurable column names and ID normalization

## Install

```bash
pip install medprep
```

`nifti_convert.load_dicom_series_as_array` (DICOM series → NIfTI) needs
`monai` and `itk`, which aren't installed by default — pull them in with:

```bash
pip install medprep[convert]
```

Or from source:

```bash
git clone https://github.com/bruno-schenberg/medprep.git
cd medprep
pip install -e .
```

## Quick example

```python
import nibabel as nib

from medprep.resample import pad_or_crop, resample_volume
from medprep.qc_checks import check_extent, check_spacing

img = nib.load("exam.nii.gz")
data, affine = img.get_fdata(dtype="float32"), img.affine

# QC: inter-slice spacing/extent from sorted slice positions (mm) along the
# acquisition axis (see qc_checks.sorted_projected_positions for DICOM input)
positions = [0.0, 0.5, 1.0, 1.5, 2.0]
spacing_result = check_spacing(positions, std_tolerance_mm=0.05)
extent_result = check_extent(positions, min_mm=50, max_mm=500)

# Resample to isotropic 1mm, then pad/crop to a fixed 128³ shape
resampled, resampled_affine = resample_volume(data, affine, target_spacing=(1.0, 1.0, 1.0))
final, final_affine = pad_or_crop(resampled, resampled_affine, target_shape=(128, 128, 128))
```

## Modules

| Module | Purpose |
|--------|---------|
| `discovery` | Archive extraction + exam folder discovery |
| `dicom_io` | DICOM header loading, original-image filtering |
| `qc_checks` | Spacing, extent, orientation, scout filtering |
| `nifti_convert` | DICOM → NIfTI + centroid XY crop |
| `resample` | Resample to target spacing/shape |
| `mip` | 3D → 2D projections (MIP, resize) |
| `visualize` | Mid-slice loaders, normalization, centroid overlay |
| `ids` | Generic sequential ID assignment |
| `metadata` | Metadata CSV joining with configurable columns |

## Requirements

Python ≥ 3.10. Key dependencies: pydicom, nibabel, numpy, pandas, scipy.
`monai` and `itk` are optional (`pip install medprep[convert]`), needed only
for `nifti_convert.load_dicom_series_as_array`.

## License

MIT
