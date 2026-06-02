"""DICOM validation: required tags, pixel data integrity, modality-specific checks.

Validates DICOM files against configurable rule sets before they enter the
preprocessing pipeline. Catches corrupt files, missing critical metadata,
and modality-specific issues (e.g., mammography-specific tag requirements)
early, preventing downstream failures.

Validation is split into three tiers:
  1. **Structural**: File can be parsed, has valid transfer syntax.
  2. **Metadata**: Required tags are present and well-formed.
  3. **Pixel data**: Pixel array dimensions match header, values are in range.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pydicom
from pydicom.dataset import Dataset
from pydicom.errors import InvalidDicomError
from pydicom.uid import (
    ExplicitVRLittleEndian,
    ImplicitVRLittleEndian,
    JPEG2000Lossless,
    JPEGBaseline8Bit,
    JPEGLosslessSV1,
    RLELossless,
)

logger = logging.getLogger(__name__)


class ValidationSeverity(Enum):
    """Severity levels for validation findings."""

    ERROR = "error"          # File cannot be processed
    WARNING = "warning"      # File can be processed but quality may be degraded
    INFO = "info"            # Informational finding


class ValidationCategory(Enum):
    """Categories of validation checks."""

    STRUCTURAL = "structural"
    METADATA = "metadata"
    PIXEL_DATA = "pixel_data"
    MODALITY_SPECIFIC = "modality_specific"


@dataclass(frozen=True)
class ValidationFinding:
    """A single finding from a validation check."""

    severity: ValidationSeverity
    category: ValidationCategory
    tag: str
    message: str
    value: Optional[Any] = None


@dataclass
class ValidationResult:
    """Result of validating a single DICOM file."""

    path: str
    is_valid: bool = True
    findings: List[ValidationFinding] = field(default_factory=list)

    @property
    def errors(self) -> List[ValidationFinding]:
        return [f for f in self.findings if f.severity == ValidationSeverity.ERROR]

    @property
    def warnings(self) -> List[ValidationFinding]:
        return [f for f in self.findings if f.severity == ValidationSeverity.WARNING]

    def add(self, finding: ValidationFinding) -> None:
        self.findings.append(finding)
        if finding.severity == ValidationSeverity.ERROR:
            self.is_valid = False


# ---------------------------------------------------------------------------
# Tag requirement definitions
# ---------------------------------------------------------------------------

# Tags required for ALL modalities
_UNIVERSAL_REQUIRED_TAGS: List[Tuple[str, str]] = [
    ("PatientID", "Patient ID"),
    ("StudyInstanceUID", "Study Instance UID"),
    ("SeriesInstanceUID", "Series Instance UID"),
    ("SOPInstanceUID", "SOP Instance UID"),
    ("SOPClassUID", "SOP Class UID"),
    ("Modality", "Modality"),
    ("Rows", "Rows"),
    ("Columns", "Columns"),
    ("BitsAllocated", "Bits Allocated"),
    ("BitsStored", "Bits Stored"),
    ("HighBit", "High Bit"),
    ("PixelRepresentation", "Pixel Representation"),
    ("SamplesPerPixel", "Samples Per Pixel"),
    ("PhotometricInterpretation", "Photometric Interpretation"),
]

# Additional tags required for mammography (MG)
_MAMMOGRAPHY_REQUIRED_TAGS: List[Tuple[str, str]] = [
    ("ImageLaterality", "Image Laterality"),
    ("ViewPosition", "View Position"),
    ("BodyPartExamined", "Body Part Examined"),
    ("Manufacturer", "Manufacturer"),
    ("PresentationIntentType", "Presentation Intent Type"),
]

# Additional tags for mammography that are strongly recommended
_MAMMOGRAPHY_RECOMMENDED_TAGS: List[Tuple[str, str]] = [
    ("WindowCenter", "Window Center"),
    ("WindowWidth", "Window Width"),
    ("VOILUTSequence", "VOI LUT Sequence"),
    ("InstitutionName", "Institution Name"),
    ("KVP", "KVP"),
    ("ExposureInuAs", "Exposure"),
    ("CompressionForce", "Compression Force"),
    ("BreastImplantPresent", "Breast Implant Present"),
    ("PaddleDescription", "Paddle Description"),
]

# Known valid transfer syntaxes
_SUPPORTED_TRANSFER_SYNTAXES: Set[str] = {
    str(ImplicitVRLittleEndian),
    str(ExplicitVRLittleEndian),
    str(JPEGBaseline8Bit),
    str(JPEGLosslessSV1),
    str(JPEG2000Lossless),
    str(RLELossless),
    "1.2.840.10008.1.2.4.80",   # JPEG-LS Lossless
    "1.2.840.10008.1.2.4.90",   # JPEG 2000 Part 1
    "1.2.840.10008.1.2.4.91",   # JPEG 2000 Part 1 Lossy
}

# Valid photometric interpretations for mammography
_VALID_MG_PHOTOMETRIC: Set[str] = {
    "MONOCHROME1", "MONOCHROME2",
}


class DicomValidator:
    """Configurable DICOM file validator.

    Runs a battery of structural, metadata, and pixel data checks. Supports
    modality-specific validation rules (mammography is the primary use case).

    Args:
        check_pixel_data: Whether to decompress and validate pixel arrays.
        modality_rules: Mapping of modality code to extra required tags.
        strict_mode: If True, warnings are promoted to errors.
        max_pixel_value_check: Whether to verify pixel value ranges.
        allowed_transfer_syntaxes: Override set of accepted transfer syntaxes.

    Example::

        validator = DicomValidator(check_pixel_data=True, strict_mode=False)
        result = validator.validate("/data/study/image001.dcm")
        if not result.is_valid:
            for err in result.errors:
                print(f"  {err.tag}: {err.message}")
    """

    def __init__(
        self,
        check_pixel_data: bool = True,
        modality_rules: Optional[Dict[str, List[Tuple[str, str]]]] = None,
        strict_mode: bool = False,
        max_pixel_value_check: bool = True,
        allowed_transfer_syntaxes: Optional[Set[str]] = None,
        custom_checks: Optional[List[Callable[[Dataset, ValidationResult], None]]] = None,
    ) -> None:
        self.check_pixel_data = check_pixel_data
        self.strict_mode = strict_mode
        self.max_pixel_value_check = max_pixel_value_check
        self.allowed_transfer_syntaxes = (
            allowed_transfer_syntaxes or _SUPPORTED_TRANSFER_SYNTAXES
        )
        self.custom_checks = custom_checks or []

        # Build modality-specific tag requirements
        self.modality_rules: Dict[str, List[Tuple[str, str]]] = {
            "MG": _MAMMOGRAPHY_REQUIRED_TAGS,
        }
        if modality_rules:
            self.modality_rules.update(modality_rules)

    def validate(self, path: Union[str, Path]) -> ValidationResult:
        """Validate a single DICOM file.

        Args:
            path: Path to the DICOM file.

        Returns:
            ValidationResult with all findings.
        """
        path = str(path)
        result = ValidationResult(path=path)

        # --- Structural checks ---
        ds = self._check_structural(path, result)
        if ds is None:
            return result

        # --- Metadata checks ---
        self._check_universal_metadata(ds, result)
        self._check_modality_specific(ds, result)

        # --- Pixel data checks ---
        if self.check_pixel_data:
            self._check_pixel_data(ds, path, result)

        # --- Custom checks ---
        for check_fn in self.custom_checks:
            try:
                check_fn(ds, result)
            except Exception as exc:
                result.add(ValidationFinding(
                    severity=ValidationSeverity.WARNING,
                    category=ValidationCategory.METADATA,
                    tag="custom_check",
                    message=f"Custom check raised: {exc}",
                ))

        return result

    def validate_batch(
        self,
        paths: List[Union[str, Path]],
    ) -> List[ValidationResult]:
        """Validate a batch of DICOM files sequentially.

        For parallel validation, use the pipeline's DistributedExecutor.

        Args:
            paths: List of file paths.

        Returns:
            List of ValidationResult objects.
        """
        return [self.validate(p) for p in paths]

    # ------------------------------------------------------------------
    # Structural checks
    # ------------------------------------------------------------------

    def _check_structural(
        self, path: str, result: ValidationResult
    ) -> Optional[Dataset]:
        """Verify the file is a parseable DICOM with valid transfer syntax."""
        # Check file exists and is readable
        p = Path(path)
        if not p.exists():
            result.add(ValidationFinding(
                severity=ValidationSeverity.ERROR,
                category=ValidationCategory.STRUCTURAL,
                tag="file",
                message=f"File does not exist: {path}",
            ))
            return None

        if p.stat().st_size < 132:
            result.add(ValidationFinding(
                severity=ValidationSeverity.ERROR,
                category=ValidationCategory.STRUCTURAL,
                tag="file_size",
                message="File too small to be valid DICOM (< 132 bytes)",
                value=p.stat().st_size,
            ))
            return None

        # Attempt parse
        try:
            ds = pydicom.dcmread(path, force=False)
        except InvalidDicomError as exc:
            result.add(ValidationFinding(
                severity=ValidationSeverity.ERROR,
                category=ValidationCategory.STRUCTURAL,
                tag="parse",
                message=f"Cannot parse DICOM: {exc}",
            ))
            return None
        except Exception as exc:
            result.add(ValidationFinding(
                severity=ValidationSeverity.ERROR,
                category=ValidationCategory.STRUCTURAL,
                tag="parse",
                message=f"Unexpected parse error: {exc}",
            ))
            return None

        # Transfer syntax
        ts_uid = ""
        if hasattr(ds, "file_meta") and ds.file_meta is not None:
            ts = getattr(ds.file_meta, "TransferSyntaxUID", None)
            ts_uid = str(ts) if ts else ""

        if ts_uid and ts_uid not in self.allowed_transfer_syntaxes:
            sev = (
                ValidationSeverity.ERROR
                if self.strict_mode
                else ValidationSeverity.WARNING
            )
            result.add(ValidationFinding(
                severity=sev,
                category=ValidationCategory.STRUCTURAL,
                tag="TransferSyntaxUID",
                message=f"Unsupported transfer syntax: {ts_uid}",
                value=ts_uid,
            ))

        return ds

    # ------------------------------------------------------------------
    # Metadata checks
    # ------------------------------------------------------------------

    def _check_universal_metadata(
        self, ds: Dataset, result: ValidationResult
    ) -> None:
        """Check that all universally required tags are present and non-empty."""
        for tag_name, display_name in _UNIVERSAL_REQUIRED_TAGS:
            val = getattr(ds, tag_name, None)
            if val is None or (isinstance(val, str) and val.strip() == ""):
                result.add(ValidationFinding(
                    severity=ValidationSeverity.ERROR,
                    category=ValidationCategory.METADATA,
                    tag=tag_name,
                    message=f"Missing required tag: {display_name}",
                ))

        # Consistency checks
        bits_alloc = getattr(ds, "BitsAllocated", 0)
        bits_stored = getattr(ds, "BitsStored", 0)
        high_bit = getattr(ds, "HighBit", 0)
        if bits_stored and bits_alloc and bits_stored > bits_alloc:
            result.add(ValidationFinding(
                severity=ValidationSeverity.ERROR,
                category=ValidationCategory.METADATA,
                tag="BitsStored",
                message=f"BitsStored ({bits_stored}) > BitsAllocated ({bits_alloc})",
            ))
        if high_bit and bits_stored and high_bit != bits_stored - 1:
            sev = (
                ValidationSeverity.ERROR
                if self.strict_mode
                else ValidationSeverity.WARNING
            )
            result.add(ValidationFinding(
                severity=sev,
                category=ValidationCategory.METADATA,
                tag="HighBit",
                message=(
                    f"HighBit ({high_bit}) != BitsStored-1 ({bits_stored - 1})"
                ),
            ))

    def _check_modality_specific(
        self, ds: Dataset, result: ValidationResult
    ) -> None:
        """Apply modality-specific validation rules."""
        modality = getattr(ds, "Modality", "")
        if not modality:
            return

        required_tags = self.modality_rules.get(modality, [])
        for tag_name, display_name in required_tags:
            val = getattr(ds, tag_name, None)
            if val is None or (isinstance(val, str) and val.strip() == ""):
                sev = (
                    ValidationSeverity.ERROR
                    if self.strict_mode
                    else ValidationSeverity.WARNING
                )
                result.add(ValidationFinding(
                    severity=sev,
                    category=ValidationCategory.MODALITY_SPECIFIC,
                    tag=tag_name,
                    message=f"Missing {modality}-required tag: {display_name}",
                ))

        # Mammography-specific value checks
        if modality == "MG":
            self._check_mammography_values(ds, result)

    def _check_mammography_values(
        self, ds: Dataset, result: ValidationResult
    ) -> None:
        """Mammography-specific value validation."""
        laterality = getattr(ds, "ImageLaterality", "")
        if laterality and laterality not in ("L", "R"):
            result.add(ValidationFinding(
                severity=ValidationSeverity.WARNING,
                category=ValidationCategory.MODALITY_SPECIFIC,
                tag="ImageLaterality",
                message=f"Unexpected laterality value: '{laterality}' (expected L or R)",
                value=laterality,
            ))

        view = getattr(ds, "ViewPosition", "")
        valid_views = {"CC", "MLO", "ML", "LM", "XCCL", "SIO", "ISO", "FB", "LMO"}
        if view and view not in valid_views:
            result.add(ValidationFinding(
                severity=ValidationSeverity.WARNING,
                category=ValidationCategory.MODALITY_SPECIFIC,
                tag="ViewPosition",
                message=f"Unusual view position: '{view}'",
                value=view,
            ))

        photometric = getattr(ds, "PhotometricInterpretation", "")
        if photometric and photometric not in _VALID_MG_PHOTOMETRIC:
            result.add(ValidationFinding(
                severity=ValidationSeverity.ERROR,
                category=ValidationCategory.MODALITY_SPECIFIC,
                tag="PhotometricInterpretation",
                message=(
                    f"Invalid photometric for mammography: '{photometric}' "
                    f"(expected MONOCHROME1 or MONOCHROME2)"
                ),
                value=photometric,
            ))

        # Check presentation intent
        intent = getattr(ds, "PresentationIntentType", "")
        if intent and intent not in ("FOR PROCESSING", "FOR PRESENTATION"):
            result.add(ValidationFinding(
                severity=ValidationSeverity.WARNING,
                category=ValidationCategory.MODALITY_SPECIFIC,
                tag="PresentationIntentType",
                message=f"Unexpected presentation intent: '{intent}'",
                value=intent,
            ))

    # ------------------------------------------------------------------
    # Pixel data checks
    # ------------------------------------------------------------------

    def _check_pixel_data(
        self, ds: Dataset, path: str, result: ValidationResult
    ) -> None:
        """Validate pixel data integrity by decompressing and checking dimensions."""
        if not hasattr(ds, "PixelData"):
            result.add(ValidationFinding(
                severity=ValidationSeverity.ERROR,
                category=ValidationCategory.PIXEL_DATA,
                tag="PixelData",
                message="No pixel data present",
            ))
            return

        try:
            pixel_array = ds.pixel_array
        except Exception as exc:
            result.add(ValidationFinding(
                severity=ValidationSeverity.ERROR,
                category=ValidationCategory.PIXEL_DATA,
                tag="PixelData",
                message=f"Cannot decompress pixel data: {exc}",
            ))
            return

        # Shape validation
        expected_rows = int(getattr(ds, "Rows", 0))
        expected_cols = int(getattr(ds, "Columns", 0))
        if expected_rows and expected_cols:
            actual_shape = pixel_array.shape
            # Handle multi-frame and multi-sample
            spatial_dims = actual_shape[-2:]
            if spatial_dims != (expected_rows, expected_cols):
                result.add(ValidationFinding(
                    severity=ValidationSeverity.ERROR,
                    category=ValidationCategory.PIXEL_DATA,
                    tag="PixelData",
                    message=(
                        f"Shape mismatch: header says ({expected_rows}, {expected_cols}) "
                        f"but array is {actual_shape}"
                    ),
                ))

        # Value range check
        if self.max_pixel_value_check:
            bits_stored = int(getattr(ds, "BitsStored", 0))
            pixel_rep = int(getattr(ds, "PixelRepresentation", 0))
            if bits_stored:
                if pixel_rep == 0:
                    max_allowed = (1 << bits_stored) - 1
                    min_allowed = 0
                else:
                    max_allowed = (1 << (bits_stored - 1)) - 1
                    min_allowed = -(1 << (bits_stored - 1))

                actual_min = int(pixel_array.min())
                actual_max = int(pixel_array.max())

                if actual_max > max_allowed or actual_min < min_allowed:
                    result.add(ValidationFinding(
                        severity=ValidationSeverity.WARNING,
                        category=ValidationCategory.PIXEL_DATA,
                        tag="PixelData",
                        message=(
                            f"Pixel values [{actual_min}, {actual_max}] exceed "
                            f"expected range [{min_allowed}, {max_allowed}] "
                            f"for {bits_stored}-bit {'signed' if pixel_rep else 'unsigned'}"
                        ),
                    ))

        # Check for all-zero or constant images
        if pixel_array.std() < 1e-6:
            result.add(ValidationFinding(
                severity=ValidationSeverity.WARNING,
                category=ValidationCategory.PIXEL_DATA,
                tag="PixelData",
                message="Pixel data is constant (zero variance) -- likely blank or corrupt",
                value=float(pixel_array.mean()),
            ))
