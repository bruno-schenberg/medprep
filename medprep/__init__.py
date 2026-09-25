"""Medical image preprocessing: DICOM discovery/conversion, resampling, QC
primitives, and 3D->2D projections.

See the module table in README.md for what lives where. This top-level
namespace re-exports one primary entry point per module for convenience;
import the submodules directly for the rest of each module's API.
"""

from .discovery import discover_exams, extract_archives
from .dicom_io import is_original, load_headers
from .ids import assign_sequential_ids
from .metadata import join_metadata
from .mip import maximum_intensity_projection
from .nifti_convert import centroid_crop_xy, load_dicom_series_as_array
from .qc_checks import check_extent, check_spacing
from .resample import pad_or_crop, resample_volume, run_resampling
from .visualize import load_mid_slice_nifti, percentile_normalize

__version__ = "0.1.0"

__all__ = [
    "assign_sequential_ids",
    "centroid_crop_xy",
    "check_extent",
    "check_spacing",
    "discover_exams",
    "extract_archives",
    "is_original",
    "join_metadata",
    "load_dicom_series_as_array",
    "load_headers",
    "load_mid_slice_nifti",
    "maximum_intensity_projection",
    "pad_or_crop",
    "percentile_normalize",
    "resample_volume",
    "run_resampling",
]
