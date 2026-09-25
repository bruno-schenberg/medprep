# Changelog

All notable changes to this project are documented in this file.
Format based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

### Fixed
- README's "Quick example" now matches the real `resample.py`/`qc_checks.py`
  APIs
- `__init__.py` populated (package docstring, `__version__`, `__all__`)
- `mip.py` now has a real `maximum_intensity_projection` function
- `pyproject.toml` gained `readme`/`license`/`classifiers`/`authors`/
  `[project.urls]` metadata
- `discovery.extract_archives` now extracts atomically (temp dir + rename),
  fixing a partial-extraction-then-skip-forever bug
- `nifti_convert.load_dicom_series_as_array` wraps the global ITK warning
  toggle in try/finally
- `resample.py`: replaced a runtime-shape `assert` with `raise ValueError`;
  collapsed a redundant `OSError` re-raise/re-catch
- Standardized error handling: `nifti_convert.resolve_dicom_file_list` now
  counts and returns unreadable DICOM headers instead of silently dropping
  them (**breaking**: now returns `(files, unreadable_count)` instead of
  just `files`)
- Replaced `print()`-as-logging with the stdlib `logging` module in
  `discovery`, `metadata`, `resample`

### Changed
- `monai`/`itk` moved to an optional `convert` extra
  (`pip install medprep[convert]`) — only needed for
  `nifti_convert.load_dicom_series_as_array`
- Dropped the unused `matplotlib` dependency
- `nifti_convert.xy_intensity_centroid` now delegates its centroid math to
  `visualize.intensity_centroid_2d` instead of duplicating it

## [0.1.0] - Initial release

Extracted from a private aneurysm-detection pipeline: DICOM discovery,
QC checks, DICOM → NIfTI conversion, resampling, 3D → 2D projections,
visualization helpers, and metadata joining.
