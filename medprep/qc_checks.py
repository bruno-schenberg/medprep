"""Composable DICOM QC primitives.

Each function operates on a list of pydicom datasets (a single series) or on
values derived from one, and returns a plain dict of fields to merge into a
per-series result. They carry no acquisition-specific assumptions (scanner
vendor, sequence type, series-selection policy) — compose them into a
pipeline and layer that policy on top in your own code.
"""
import statistics

from .dicom_io import is_original


def group_by_series_uid(datasets: list) -> dict:
    """Group DICOM datasets by SeriesInstanceUID. Missing UIDs group under None."""
    groups: dict = {}
    for ds in datasets:
        uid = getattr(ds, "SeriesInstanceUID", None)
        groups.setdefault(uid, []).append(ds)
    return groups


def filter_scouts(datasets: list) -> tuple[list, dict]:
    """Remove scout/derived slices. Returns (original_slices, field_updates)."""
    original = [ds for ds in datasets if is_original(ds)]
    scout_count = len(datasets) - len(original)
    return original, {
        "scout_slice_count": scout_count,
        "scout_slices": scout_count > 0,
        "slice_issue": "no_originals" if len(original) == 0 else None,
    }


def read_series_tags(original: list) -> dict:
    """Extract slice count, modality, and series identification tags from the
    first slice of a series."""
    first = original[0]
    fields = {"slice_count": len(original)}
    try:
        fields["modality"] = str(first.Modality)
    except AttributeError:
        pass
    for attr, col in [
        ("SeriesNumber", "series_number"),
        ("SeriesDescription", "series_description"),
        ("ProtocolName", "protocol_name"),
        ("MRAcquisitionType", "mra_acquisition_type"),
        ("ScanningSequence", "scanning_sequence"),
        ("ScanOptions", "scan_options"),
    ]:
        try:
            fields[col] = str(getattr(first, attr))
        except AttributeError:
            pass
    return fields


def compute_slice_normal(iop) -> tuple[float, float, float] | None:
    """Slice normal = cross product of the two ImageOrientationPatient direction cosines.

    Returns None if iop is malformed (wrong length or non-numeric values).
    """
    try:
        r = [float(iop[i]) for i in range(3)]
        c = [float(iop[i]) for i in range(3, 6)]
        return (
            r[1] * c[2] - r[2] * c[1],
            r[2] * c[0] - r[0] * c[2],
            r[0] * c[1] - r[1] * c[0],
        )
    except (IndexError, ValueError, TypeError):
        return None


def classify_orientation(normal: tuple[float, float, float]) -> str:
    """Classify slice plane from the dominant axis of the normal vector.
    x-dominant -> sagittal, y-dominant -> coronal, z-dominant -> axial."""
    ax, ay, az = abs(normal[0]), abs(normal[1]), abs(normal[2])
    m = max(ax, ay, az)
    if az == m:
        return "axial"
    if ay == m:
        return "coronal"
    return "sagittal"


def project_position(ds, normal: tuple[float, float, float]) -> float | None:
    """Project ImagePositionPatient onto the normal vector.

    Using projection (not raw z) is correct for oblique acquisitions where the
    imaging plane is tilted relative to the standard anatomical axes.
    """
    try:
        ipp = ds.ImagePositionPatient
        return sum(float(ipp[i]) * normal[i] for i in range(3))
    except (AttributeError, IndexError, ValueError):
        return None


def check_spatial_metadata(original: list) -> tuple[dict, tuple | None]:
    """
    Verify IOP/IPP/PixelSpacing on the first slice and compute the slice normal.

    Returns (field_updates, normal).
    normal is None when spatial metadata is absent or malformed — the caller
    should treat this as missing_spatial_metadata and stop further checks.
    """
    first = original[0]
    try:
        iop = first.ImageOrientationPatient
        _ = first.ImagePositionPatient
        _ = first.PixelSpacing
    except AttributeError:
        return {"missing_spatial_metadata": True}, None

    normal = compute_slice_normal(iop)
    if normal is None:
        return {"missing_spatial_metadata": True}, None

    fields = {
        "missing_spatial_metadata": False,
        "orientation": classify_orientation(normal),
    }
    try:
        fields["rows"] = int(first.Rows)
        fields["columns"] = int(first.Columns)
    except AttributeError:
        pass
    try:
        fields["voxel_spacing_row"] = float(first.PixelSpacing[0])
        fields["voxel_spacing_col"] = float(first.PixelSpacing[1])
    except (AttributeError, IndexError, ValueError):
        pass

    return fields, normal


def sorted_projected_positions(original: list, normal: tuple) -> list[float] | None:
    """
    Project each slice's ImagePositionPatient onto the normal vector and sort.

    Returns sorted positions, or None if any slice is missing IPP.
    Sorting by projection (not raw z) is correct for oblique acquisitions.
    """
    positions = []
    for ds in original:
        pos = project_position(ds, normal)
        if pos is None:
            return None
        positions.append(pos)
    return sorted(positions)


def check_spacing(positions: list[float], std_tolerance_mm: float) -> dict:
    """Evaluate inter-slice gaps for duplicates, missing slices, and consistency.

    positions: sorted slice positions (mm) along the slice normal.
    std_tolerance_mm: gaps with stddev above this are flagged inconsistent_spacing.
    """
    gaps = [positions[i + 1] - positions[i] for i in range(len(positions) - 1)]
    median_gap = statistics.median(gaps)
    std_gap = statistics.pstdev(gaps)

    slice_issues = []
    if any(g < 0.001 for g in gaps):
        slice_issues.append("duplicates")
    if any(g > 1.5 * median_gap for g in gaps):
        slice_issues.append("missing_slices")

    return {
        "voxel_spacing_z": round(median_gap, 6),
        "slice_issue": ",".join(slice_issues) if slice_issues else None,
        "inconsistent_spacing": std_gap > std_tolerance_mm,
    }


def check_extent(positions: list[float], min_mm: float, max_mm: float) -> dict:
    """Compute physical z-span and check against the given limits.

    extent_z_mm = positions[-1] - positions[0]: distance between the first and
    last slice along the acquisition direction. Not slice_count × z_spacing —
    that formula counts one extra interval.
    Example: 254 slices at 0.5mm -> extent = 253 × 0.5 = 126.5mm (not 127.0mm).
    """
    extent = positions[-1] - positions[0]
    if extent < min_mm:
        extent_issue = "too_short"
    elif extent > max_mm:
        extent_issue = "too_long"
    else:
        extent_issue = None
    return {
        "extent_z_mm": round(extent, 3),
        "extent_issue": extent_issue,
    }
