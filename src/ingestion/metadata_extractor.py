"""Extract and normalize DICOM metadata into structured format.

Handles manufacturer-specific tag variations across Hologic, GE Healthcare,
and Siemens Healthineers mammography systems. Produces a flat, normalized
dictionary suitable for Parquet serialization and downstream ML feature use.

Manufacturer-specific handling includes:
  - Hologic: Selenia/Dimensions private tags for paddle info, compression
  - GE: Senographe private tags for AEC mode, target/filter
  - Siemens: Mammomat private tags for exposure parameters
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import pydicom
from pydicom.dataset import Dataset
from pydicom.tag import Tag

logger = logging.getLogger(__name__)

# Manufacturer canonical names (DICOM Manufacturer field is free-text)
_MANUFACTURER_ALIASES: Dict[str, str] = {
    "hologic": "Hologic",
    "hologic, inc": "Hologic",
    "hologic, inc.": "Hologic",
    "ge": "GE Healthcare",
    "ge healthcare": "GE Healthcare",
    "ge medical systems": "GE Healthcare",
    "ge healthcare technologies": "GE Healthcare",
    "siemens": "Siemens Healthineers",
    "siemens healthineers": "Siemens Healthineers",
    "siemens healthcare gmbh": "Siemens Healthineers",
    "fuji": "Fujifilm",
    "fujifilm": "Fujifilm",
    "fujifilm corporation": "Fujifilm",
    "philips": "Philips",
    "philips healthcare": "Philips",
    "philips medical systems": "Philips",
}

# Hologic private tags (group 0x7E01)
_HOLOGIC_PRIVATE_TAGS = {
    "PaddleDescription": Tag(0x7E01, 0x1002),
    "CompressionThickness": Tag(0x7E01, 0x1006),
    "PaddleID": Tag(0x7E01, 0x100A),
    "ExposureMode": Tag(0x7E01, 0x1018),
}

# GE private tags (group 0x0045)
_GE_PRIVATE_TAGS = {
    "AECMode": Tag(0x0045, 0x101B),
    "TargetMaterial": Tag(0x0045, 0x101D),
    "FilterMaterial": Tag(0x0045, 0x101E),
    "Thickness": Tag(0x0045, 0x101A),
}


@dataclass
class NormalizedMetadata:
    """Flat, normalized metadata record for a single DICOM image.

    All fields are typed and ready for DataFrame / Parquet serialization.
    Optional fields default to None, which maps to null in Parquet.
    """

    # Identity
    sop_instance_uid: str = ""
    study_instance_uid: str = ""
    series_instance_uid: str = ""
    patient_id: str = ""
    accession_number: str = ""

    # Demographics (de-identified)
    patient_age_years: Optional[int] = None
    patient_sex: Optional[str] = None

    # Study info
    study_date: Optional[str] = None  # ISO 8601 YYYY-MM-DD
    study_description: Optional[str] = None
    institution_name: Optional[str] = None
    referring_physician: Optional[str] = None

    # Series / Image info
    modality: str = ""
    series_description: Optional[str] = None
    series_number: Optional[int] = None
    instance_number: Optional[int] = None
    image_type: Optional[str] = None  # Joined backslash-separated

    # Acquisition parameters
    manufacturer: str = ""
    manufacturer_normalized: str = ""
    manufacturer_model: Optional[str] = None
    software_version: Optional[str] = None
    station_name: Optional[str] = None
    detector_type: Optional[str] = None

    # Image geometry
    rows: int = 0
    columns: int = 0
    pixel_spacing_mm: Optional[Tuple[float, float]] = None
    bits_allocated: int = 0
    bits_stored: int = 0
    pixel_representation: int = 0  # 0=unsigned, 1=signed
    photometric_interpretation: str = ""
    samples_per_pixel: int = 1
    transfer_syntax_uid: str = ""

    # Windowing
    window_center: Optional[float] = None
    window_width: Optional[float] = None
    rescale_intercept: float = 0.0
    rescale_slope: float = 1.0
    has_voi_lut: bool = False

    # Mammography-specific
    image_laterality: Optional[str] = None
    view_position: Optional[str] = None
    body_part: Optional[str] = None
    presentation_intent: Optional[str] = None
    breast_implant_present: Optional[bool] = None
    compression_force_n: Optional[float] = None
    compression_thickness_mm: Optional[float] = None
    kvp: Optional[float] = None
    exposure_uas: Optional[float] = None
    anode_target: Optional[str] = None
    filter_material: Optional[str] = None
    paddle_description: Optional[str] = None

    # Derived / computed
    aspect_ratio: Optional[float] = None
    megapixels: Optional[float] = None

    # Quality flags
    has_pixel_data: bool = False
    is_for_processing: bool = False
    is_for_presentation: bool = False
    manufacturer_private_tags_extracted: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to flat dictionary for DataFrame construction."""
        d: Dict[str, Any] = {}
        for k, v in self.__dict__.items():
            if isinstance(v, tuple):
                # Flatten tuples (e.g. pixel_spacing_mm)
                for i, val in enumerate(v):
                    d[f"{k}_{i}"] = val
            else:
                d[k] = v
        return d


def _normalize_manufacturer(raw: str) -> str:
    """Map free-text manufacturer string to canonical name."""
    if not raw:
        return "Unknown"
    key = raw.strip().lower()
    for alias, canonical in _MANUFACTURER_ALIASES.items():
        if alias in key:
            return canonical
    return raw.strip()


def _parse_date(date_str: str) -> Optional[str]:
    """Parse DICOM date (DA VR: YYYYMMDD) to ISO 8601."""
    if not date_str or not date_str.strip():
        return None
    clean = re.sub(r"[^0-9]", "", date_str.strip())
    if len(clean) == 8:
        try:
            dt = datetime.strptime(clean, "%Y%m%d")
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            return None
    return None


def _parse_age(age_str: str) -> Optional[int]:
    """Parse DICOM age string (e.g., '065Y') to integer years."""
    if not age_str:
        return None
    match = re.match(r"(\d+)([YMWD])?", age_str.strip())
    if not match:
        return None
    value = int(match.group(1))
    unit = match.group(2) or "Y"
    if unit == "Y":
        return value
    elif unit == "M":
        return value // 12
    elif unit == "W":
        return value // 52
    elif unit == "D":
        return value // 365
    return value


def _safe_float(val: Any) -> Optional[float]:
    """Safely convert a DICOM value to float."""
    if val is None:
        return None
    try:
        if isinstance(val, (list, pydicom.multival.MultiValue)):
            return float(val[0])
        return float(val)
    except (ValueError, TypeError, IndexError):
        return None


def _safe_int(val: Any) -> Optional[int]:
    """Safely convert a DICOM value to int."""
    if val is None:
        return None
    try:
        if isinstance(val, (list, pydicom.multival.MultiValue)):
            return int(val[0])
        return int(val)
    except (ValueError, TypeError, IndexError):
        return None


def _safe_str(val: Any) -> Optional[str]:
    """Safely convert a DICOM value to string."""
    if val is None:
        return None
    s = str(val).strip()
    return s if s else None


def _get_private_tag(ds: Dataset, tag: Tag) -> Optional[Any]:
    """Safely retrieve a private DICOM tag value."""
    try:
        elem = ds[tag]
        return elem.value if elem else None
    except (KeyError, IndexError):
        return None


class MetadataExtractor:
    """Extract and normalize DICOM metadata with manufacturer harmonization.

    Handles the substantial variation in how different manufacturers encode
    equivalent clinical information. Produces a flat NormalizedMetadata
    record suitable for tabular storage and ML feature engineering.

    Args:
        extract_private_tags: Whether to attempt private tag extraction.
        normalize_dates: Whether to convert dates to ISO 8601.
        compute_derived: Whether to compute derived fields (megapixels, etc.).

    Example::

        extractor = MetadataExtractor(extract_private_tags=True)
        ds = pydicom.dcmread("mammogram.dcm", stop_before_pixels=True)
        meta = extractor.extract(ds)
        print(meta.manufacturer_normalized)  # "Hologic"
        print(meta.pixel_spacing_mm)         # (0.07, 0.07)
    """

    def __init__(
        self,
        extract_private_tags: bool = True,
        normalize_dates: bool = True,
        compute_derived: bool = True,
    ) -> None:
        self.extract_private_tags = extract_private_tags
        self.normalize_dates = normalize_dates
        self.compute_derived = compute_derived

    def extract(self, ds: Dataset) -> NormalizedMetadata:
        """Extract normalized metadata from a pydicom Dataset.

        Args:
            ds: Parsed DICOM dataset (can be read with stop_before_pixels=True).

        Returns:
            NormalizedMetadata with all available fields populated.
        """
        meta = NormalizedMetadata()

        # --- Identity ---
        meta.sop_instance_uid = _safe_str(getattr(ds, "SOPInstanceUID", "")) or ""
        meta.study_instance_uid = _safe_str(getattr(ds, "StudyInstanceUID", "")) or ""
        meta.series_instance_uid = _safe_str(getattr(ds, "SeriesInstanceUID", "")) or ""
        meta.patient_id = _safe_str(getattr(ds, "PatientID", "")) or ""
        meta.accession_number = _safe_str(getattr(ds, "AccessionNumber", "")) or ""

        # --- Demographics ---
        meta.patient_age_years = _parse_age(_safe_str(getattr(ds, "PatientAge", "")) or "")
        meta.patient_sex = _safe_str(getattr(ds, "PatientSex", ""))

        # --- Study ---
        raw_date = _safe_str(getattr(ds, "StudyDate", "")) or ""
        meta.study_date = _parse_date(raw_date) if self.normalize_dates else raw_date
        meta.study_description = _safe_str(getattr(ds, "StudyDescription", ""))
        meta.institution_name = _safe_str(getattr(ds, "InstitutionName", ""))
        meta.referring_physician = _safe_str(getattr(ds, "ReferringPhysicianName", ""))

        # --- Series / Image ---
        meta.modality = _safe_str(getattr(ds, "Modality", "")) or ""
        meta.series_description = _safe_str(getattr(ds, "SeriesDescription", ""))
        meta.series_number = _safe_int(getattr(ds, "SeriesNumber", None))
        meta.instance_number = _safe_int(getattr(ds, "InstanceNumber", None))
        image_type = getattr(ds, "ImageType", None)
        if image_type is not None:
            if isinstance(image_type, (list, pydicom.multival.MultiValue)):
                meta.image_type = "\\".join(str(x) for x in image_type)
            else:
                meta.image_type = str(image_type)

        # --- Acquisition ---
        raw_mfr = _safe_str(getattr(ds, "Manufacturer", "")) or ""
        meta.manufacturer = raw_mfr
        meta.manufacturer_normalized = _normalize_manufacturer(raw_mfr)
        meta.manufacturer_model = _safe_str(getattr(ds, "ManufacturerModelName", ""))
        meta.software_version = _safe_str(getattr(ds, "SoftwareVersions", ""))
        meta.station_name = _safe_str(getattr(ds, "StationName", ""))
        meta.detector_type = _safe_str(getattr(ds, "DetectorType", ""))

        # --- Image geometry ---
        meta.rows = _safe_int(getattr(ds, "Rows", 0)) or 0
        meta.columns = _safe_int(getattr(ds, "Columns", 0)) or 0
        meta.bits_allocated = _safe_int(getattr(ds, "BitsAllocated", 0)) or 0
        meta.bits_stored = _safe_int(getattr(ds, "BitsStored", 0)) or 0
        meta.pixel_representation = _safe_int(getattr(ds, "PixelRepresentation", 0)) or 0
        meta.photometric_interpretation = (
            _safe_str(getattr(ds, "PhotometricInterpretation", "")) or ""
        )
        meta.samples_per_pixel = _safe_int(getattr(ds, "SamplesPerPixel", 1)) or 1

        # Pixel spacing
        ps = getattr(ds, "PixelSpacing", None)
        if ps is None:
            ps = getattr(ds, "ImagerPixelSpacing", None)
        if ps is not None:
            try:
                meta.pixel_spacing_mm = (float(ps[0]), float(ps[1]))
            except (IndexError, TypeError, ValueError):
                pass

        # Transfer syntax
        if hasattr(ds, "file_meta") and ds.file_meta is not None:
            ts = getattr(ds.file_meta, "TransferSyntaxUID", None)
            meta.transfer_syntax_uid = str(ts) if ts else ""

        # --- Windowing ---
        meta.window_center = _safe_float(getattr(ds, "WindowCenter", None))
        meta.window_width = _safe_float(getattr(ds, "WindowWidth", None))
        meta.rescale_intercept = _safe_float(getattr(ds, "RescaleIntercept", 0.0)) or 0.0
        meta.rescale_slope = _safe_float(getattr(ds, "RescaleSlope", 1.0)) or 1.0
        meta.has_voi_lut = hasattr(ds, "VOILUTSequence") and ds.VOILUTSequence is not None

        # --- Mammography-specific ---
        meta.image_laterality = _safe_str(getattr(ds, "ImageLaterality", ""))
        if not meta.image_laterality:
            meta.image_laterality = _safe_str(getattr(ds, "Laterality", ""))
        meta.view_position = _safe_str(getattr(ds, "ViewPosition", ""))
        meta.body_part = _safe_str(getattr(ds, "BodyPartExamined", ""))
        meta.presentation_intent = _safe_str(getattr(ds, "PresentationIntentType", ""))
        meta.is_for_processing = meta.presentation_intent == "FOR PROCESSING"
        meta.is_for_presentation = meta.presentation_intent == "FOR PRESENTATION"

        implant = getattr(ds, "BreastImplantPresent", None)
        if implant is not None:
            meta.breast_implant_present = str(implant).upper() in ("YES", "Y", "1")

        meta.compression_force_n = _safe_float(getattr(ds, "CompressionForce", None))
        meta.kvp = _safe_float(getattr(ds, "KVP", None))
        meta.exposure_uas = _safe_float(getattr(ds, "ExposureInuAs", None))
        if meta.exposure_uas is None:
            exposure_mas = _safe_float(getattr(ds, "Exposure", None))
            if exposure_mas is not None:
                meta.exposure_uas = exposure_mas * 1000.0

        meta.anode_target = _safe_str(getattr(ds, "AnodeTargetMaterial", ""))
        meta.filter_material = _safe_str(getattr(ds, "FilterMaterial", ""))
        meta.paddle_description = _safe_str(getattr(ds, "PaddleDescription", ""))

        # --- Private tags ---
        if self.extract_private_tags and meta.modality == "MG":
            self._extract_manufacturer_private(ds, meta)

        # --- Pixel data flag ---
        meta.has_pixel_data = hasattr(ds, "PixelData") and ds.PixelData is not None

        # --- Derived ---
        if self.compute_derived:
            if meta.rows > 0 and meta.columns > 0:
                meta.megapixels = round(meta.rows * meta.columns / 1e6, 2)
                meta.aspect_ratio = round(meta.columns / meta.rows, 3)

        return meta

    def extract_from_file(self, path: str) -> NormalizedMetadata:
        """Convenience: read a DICOM file and extract metadata.

        Args:
            path: Path to DICOM file.

        Returns:
            NormalizedMetadata.
        """
        ds = pydicom.dcmread(path, stop_before_pixels=True, force=True)
        return self.extract(ds)

    def _extract_manufacturer_private(
        self, ds: Dataset, meta: NormalizedMetadata
    ) -> None:
        """Extract manufacturer-specific private tags for mammography."""
        mfr = meta.manufacturer_normalized

        if mfr == "Hologic":
            self._extract_hologic_private(ds, meta)
        elif mfr == "GE Healthcare":
            self._extract_ge_private(ds, meta)
        elif mfr == "Siemens Healthineers":
            self._extract_siemens_private(ds, meta)

        meta.manufacturer_private_tags_extracted = True

    def _extract_hologic_private(
        self, ds: Dataset, meta: NormalizedMetadata
    ) -> None:
        """Hologic Selenia/Dimensions private tag extraction."""
        paddle = _get_private_tag(ds, _HOLOGIC_PRIVATE_TAGS["PaddleDescription"])
        if paddle and meta.paddle_description is None:
            meta.paddle_description = str(paddle)

        thickness = _get_private_tag(ds, _HOLOGIC_PRIVATE_TAGS["CompressionThickness"])
        if thickness and meta.compression_thickness_mm is None:
            meta.compression_thickness_mm = _safe_float(thickness)

        logger.debug("Hologic private tags extracted for %s", meta.sop_instance_uid)

    def _extract_ge_private(
        self, ds: Dataset, meta: NormalizedMetadata
    ) -> None:
        """GE Senographe private tag extraction."""
        target = _get_private_tag(ds, _GE_PRIVATE_TAGS["TargetMaterial"])
        if target and meta.anode_target is None:
            meta.anode_target = str(target)

        filt = _get_private_tag(ds, _GE_PRIVATE_TAGS["FilterMaterial"])
        if filt and meta.filter_material is None:
            meta.filter_material = str(filt)

        thickness = _get_private_tag(ds, _GE_PRIVATE_TAGS["Thickness"])
        if thickness and meta.compression_thickness_mm is None:
            meta.compression_thickness_mm = _safe_float(thickness)

        logger.debug("GE private tags extracted for %s", meta.sop_instance_uid)

    def _extract_siemens_private(
        self, ds: Dataset, meta: NormalizedMetadata
    ) -> None:
        """Siemens Mammomat private tag extraction.

        Siemens stores breast thickness and compression data in group 0x0019
        and dose information in group 0x0021 on Mammomat systems.
        """
        # Siemens private group 0x0019 for compression data
        try:
            elem = ds[Tag(0x0019, 0x10BB)]
            if elem and meta.compression_thickness_mm is None:
                meta.compression_thickness_mm = _safe_float(elem.value)
        except (KeyError, IndexError):
            pass

        logger.debug("Siemens private tags extracted for %s", meta.sop_instance_uid)


def extract_batch(
    paths: List[str],
    extract_private_tags: bool = True,
) -> List[Dict[str, Any]]:
    """Extract normalized metadata from a batch of DICOM files.

    Convenience function for bulk extraction. Returns list of dicts
    suitable for ``pandas.DataFrame`` construction.

    Args:
        paths: List of DICOM file paths.
        extract_private_tags: Whether to extract vendor private tags.

    Returns:
        List of flat dictionaries, one per file.
    """
    extractor = MetadataExtractor(extract_private_tags=extract_private_tags)
    results: List[Dict[str, Any]] = []
    for path in paths:
        try:
            meta = extractor.extract_from_file(path)
            d = meta.to_dict()
            d["source_path"] = path
            d["extraction_error"] = None
            results.append(d)
        except Exception as exc:
            results.append({
                "source_path": path,
                "extraction_error": str(exc),
            })
            logger.warning("Metadata extraction failed for %s: %s", path, exc)
    return results
