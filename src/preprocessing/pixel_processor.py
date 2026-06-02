"""Pixel data processing for DICOM images.

Handles the critical and often error-prone step of converting raw DICOM pixel
data into normalized floating-point arrays suitable for ML consumption. This
includes:

  - VOI LUT (Value of Interest Look-Up Table) application
  - Modality LUT / Rescale Slope+Intercept application
  - Photometric interpretation handling (MONOCHROME1 vs MONOCHROME2 inversion)
  - Bit depth normalization (8/10/12/14/16-bit to float32)
  - Manufacturer-specific pixel corrections
  - Presentation state handling

The correct order of operations follows the DICOM standard display pipeline:
  Raw Stored Values -> Modality LUT -> VOI LUT -> Presentation LUT

References:
  - DICOM PS3.3 C.11.1 (Modality LUT)
  - DICOM PS3.3 C.11.2 (VOI LUT)
  - DICOM PS3.3 C.7.6.3.1.2 (Photometric Interpretation)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pydicom
from pydicom.dataset import Dataset

logger = logging.getLogger(__name__)


class PhotometricInterpretation(Enum):
    """Standard DICOM photometric interpretations for grayscale images."""

    MONOCHROME1 = "MONOCHROME1"  # 0 = white (inverted)
    MONOCHROME2 = "MONOCHROME2"  # 0 = black  (standard)


class NormalizationMethod(Enum):
    """Methods for normalizing pixel intensities to [0, 1] range."""

    MIN_MAX = "min_max"                # Per-image min-max
    PERCENTILE = "percentile"          # Clip to [p1, p99] then min-max
    FIXED_RANGE = "fixed_range"        # Use known bit depth range
    HISTOGRAM_EQ = "histogram_eq"      # Histogram equalization
    Z_SCORE = "z_score"                # Zero mean, unit variance


@dataclass
class PixelProcessingConfig:
    """Configuration for pixel data processing."""

    apply_modality_lut: bool = True
    apply_voi_lut: bool = True
    target_photometric: PhotometricInterpretation = PhotometricInterpretation.MONOCHROME2
    normalization: NormalizationMethod = NormalizationMethod.PERCENTILE
    percentile_low: float = 0.5
    percentile_high: float = 99.5
    output_dtype: str = "float32"
    output_bit_depth: Optional[int] = None  # None = float, 8/16 = uint
    clip_negative: bool = True
    apply_manufacturer_corrections: bool = True
    fixed_range_min: float = 0.0
    fixed_range_max: float = 4095.0  # 12-bit default


# ---------------------------------------------------------------------------
# Manufacturer-specific correction profiles
# ---------------------------------------------------------------------------

@dataclass
class ManufacturerProfile:
    """Manufacturer-specific pixel correction parameters."""

    name: str
    # Some manufacturers store raw detector values with a known offset
    pixel_offset: float = 0.0
    # Some systems have known dead pixel columns
    dead_column_indices: Optional[List[int]] = None
    # Hologic Selenia stores paddle-attenuation pixels that need masking
    mask_paddle_region: bool = False
    # GE Senographe has a specific log transform for raw data
    apply_log_transform: bool = False
    log_transform_base: float = 10.0


_MANUFACTURER_PROFILES: Dict[str, ManufacturerProfile] = {
    "Hologic": ManufacturerProfile(
        name="Hologic",
        pixel_offset=0.0,
        mask_paddle_region=True,
    ),
    "GE Healthcare": ManufacturerProfile(
        name="GE Healthcare",
        pixel_offset=0.0,
        apply_log_transform=False,
    ),
    "Siemens Healthineers": ManufacturerProfile(
        name="Siemens Healthineers",
        pixel_offset=0.0,
    ),
}


def apply_modality_lut(
    pixel_array: np.ndarray,
    ds: Dataset,
) -> np.ndarray:
    """Apply Modality LUT transformation (Rescale Slope/Intercept or LUT Sequence).

    Converts stored pixel values to modality-specific units (e.g., Hounsfield
    units for CT, optical density for mammography).

    If a Modality LUT Sequence is present, it takes precedence over
    RescaleSlope/RescaleIntercept per the DICOM standard.

    Args:
        pixel_array: Raw stored pixel values as numpy array.
        ds: pydicom Dataset with LUT information.

    Returns:
        Transformed pixel array as float64.
    """
    output = pixel_array.astype(np.float64)

    # Check for Modality LUT Sequence first (takes precedence)
    if hasattr(ds, "ModalityLUTSequence") and ds.ModalityLUTSequence:
        lut_seq = ds.ModalityLUTSequence[0]
        lut_data = np.array(lut_seq.LUTData, dtype=np.float64)
        descriptor = lut_seq.LUTDescriptor
        n_entries = descriptor[0] if descriptor[0] != 0 else 65536
        first_value = descriptor[1]

        # Apply LUT via lookup
        indices = np.clip(
            output.astype(np.int64) - first_value, 0, n_entries - 1
        ).astype(np.int64)
        output = lut_data[indices]
        logger.debug("Applied Modality LUT Sequence (%d entries)", n_entries)
        return output

    # Fall back to Rescale Slope/Intercept
    slope = float(getattr(ds, "RescaleSlope", 1.0) or 1.0)
    intercept = float(getattr(ds, "RescaleIntercept", 0.0) or 0.0)

    if slope != 1.0 or intercept != 0.0:
        output = output * slope + intercept
        logger.debug("Applied RescaleSlope=%.4f, RescaleIntercept=%.4f", slope, intercept)

    return output


def apply_voi_lut(
    pixel_array: np.ndarray,
    ds: Dataset,
    window_index: int = 0,
) -> np.ndarray:
    """Apply VOI LUT transformation (Window Center/Width or VOI LUT Sequence).

    Maps modality pixel values to display-ready values. For mammography,
    this is critical because raw "FOR PROCESSING" images often have extreme
    dynamic range that needs windowing for useful ML features.

    Args:
        pixel_array: Modality-transformed pixel values.
        ds: pydicom Dataset.
        window_index: Which window to apply if multiple are defined.

    Returns:
        Windowed pixel array, mapped to output range [0, max_out].
    """
    output = pixel_array.astype(np.float64)

    # Check for VOI LUT Sequence
    if hasattr(ds, "VOILUTSequence") and ds.VOILUTSequence:
        try:
            lut_seq = ds.VOILUTSequence[window_index]
            lut_data = np.array(lut_seq.LUTData, dtype=np.float64)
            descriptor = lut_seq.LUTDescriptor
            n_entries = descriptor[0] if descriptor[0] != 0 else 65536
            first_value = descriptor[1]

            indices = np.clip(
                output.astype(np.int64) - first_value, 0, n_entries - 1
            ).astype(np.int64)
            output = lut_data[indices]
            logger.debug("Applied VOI LUT Sequence (%d entries)", n_entries)
            return output
        except (IndexError, AttributeError) as exc:
            logger.warning("VOI LUT Sequence application failed: %s", exc)

    # Fall back to Window Center/Width
    wc = getattr(ds, "WindowCenter", None)
    ww = getattr(ds, "WindowWidth", None)

    if wc is not None and ww is not None:
        # Handle multi-valued
        if isinstance(wc, pydicom.multival.MultiValue):
            wc = float(wc[min(window_index, len(wc) - 1)])
        else:
            wc = float(wc)
        if isinstance(ww, pydicom.multival.MultiValue):
            ww = float(ww[min(window_index, len(ww) - 1)])
        else:
            ww = float(ww)

        if ww <= 0:
            logger.warning("Window Width <= 0 (%.1f), skipping VOI", ww)
            return output

        # Linear VOI LUT function (DICOM PS3.3 C.11.2.1.2.1)
        lower = wc - ww / 2.0
        upper = wc + ww / 2.0
        output = np.clip(output, lower, upper)
        output = (output - lower) / (upper - lower)
        logger.debug("Applied Window Center=%.1f, Width=%.1f", wc, ww)
        return output

    logger.debug("No VOI LUT information available; returning unwindowed data")
    return output


def handle_photometric_interpretation(
    pixel_array: np.ndarray,
    source: str,
    target: PhotometricInterpretation = PhotometricInterpretation.MONOCHROME2,
) -> np.ndarray:
    """Normalize photometric interpretation.

    MONOCHROME1 means minimum pixel value = white (used in some CR/DR systems).
    MONOCHROME2 means minimum pixel value = black (standard for most modalities).
    For ML, we always want MONOCHROME2 so that higher values = denser tissue.

    Args:
        pixel_array: Input pixel array.
        source: Source photometric interpretation string.
        target: Desired output interpretation.

    Returns:
        Pixel array with correct photometric interpretation.
    """
    source_is_m1 = source.strip().upper() == "MONOCHROME1"
    target_is_m1 = target == PhotometricInterpretation.MONOCHROME1

    if source_is_m1 != target_is_m1:
        # Need inversion
        if np.issubdtype(pixel_array.dtype, np.floating):
            pixel_array = pixel_array.max() - pixel_array
        else:
            info = np.iinfo(pixel_array.dtype)
            pixel_array = info.max - pixel_array
        logger.debug("Inverted photometric: %s -> %s", source, target.value)

    return pixel_array


def normalize_bit_depth(
    pixel_array: np.ndarray,
    bits_stored: int,
    pixel_representation: int = 0,
    config: Optional[PixelProcessingConfig] = None,
) -> np.ndarray:
    """Normalize pixel values based on stored bit depth and normalization method.

    Args:
        pixel_array: Input array (any dtype).
        bits_stored: DICOM BitsStored value.
        pixel_representation: 0=unsigned, 1=signed.
        config: Processing configuration.

    Returns:
        Normalized float32 array.
    """
    if config is None:
        config = PixelProcessingConfig()

    arr = pixel_array.astype(np.float32)

    if config.clip_negative and pixel_representation == 0:
        arr = np.clip(arr, 0, None)

    method = config.normalization

    if method == NormalizationMethod.FIXED_RANGE:
        if pixel_representation == 0:
            max_val = float((1 << bits_stored) - 1)
            arr = arr / max_val
        else:
            half = float(1 << (bits_stored - 1))
            arr = (arr + half) / (2 * half)

    elif method == NormalizationMethod.MIN_MAX:
        vmin, vmax = float(arr.min()), float(arr.max())
        if vmax > vmin:
            arr = (arr - vmin) / (vmax - vmin)
        else:
            arr = np.zeros_like(arr)

    elif method == NormalizationMethod.PERCENTILE:
        p_low = np.percentile(arr, config.percentile_low)
        p_high = np.percentile(arr, config.percentile_high)
        if p_high > p_low:
            arr = np.clip(arr, p_low, p_high)
            arr = (arr - p_low) / (p_high - p_low)
        else:
            arr = np.zeros_like(arr)

    elif method == NormalizationMethod.Z_SCORE:
        mean = arr.mean()
        std = arr.std()
        if std > 1e-8:
            arr = (arr - mean) / std
        else:
            arr = np.zeros_like(arr)

    elif method == NormalizationMethod.HISTOGRAM_EQ:
        arr = _histogram_equalize(arr)

    return arr


def _histogram_equalize(arr: np.ndarray, n_bins: int = 4096) -> np.ndarray:
    """Apply histogram equalization to a float array.

    Args:
        arr: Input float32 array.
        n_bins: Number of histogram bins.

    Returns:
        Equalized float32 array in [0, 1].
    """
    flat = arr.flatten()
    hist, bin_edges = np.histogram(flat, bins=n_bins, density=False)
    cdf = hist.cumsum().astype(np.float64)
    cdf_min = cdf[cdf > 0].min() if (cdf > 0).any() else 0
    cdf_range = cdf[-1] - cdf_min
    if cdf_range == 0:
        return np.zeros_like(arr)
    cdf_normalized = (cdf - cdf_min) / cdf_range
    # Map pixel values to CDF
    indices = np.clip(
        np.digitize(flat, bin_edges[:-1]) - 1, 0, n_bins - 1
    )
    equalized = cdf_normalized[indices].reshape(arr.shape)
    return equalized.astype(np.float32)


class PixelProcessor:
    """Full pixel data processing pipeline for DICOM images.

    Orchestrates the complete pixel transformation chain:
    Raw -> Modality LUT -> VOI LUT -> Photometric -> Normalize -> Output

    Args:
        config: Processing configuration.
        manufacturer_profiles: Override manufacturer correction profiles.

    Example::

        processor = PixelProcessor(PixelProcessingConfig(
            normalization=NormalizationMethod.PERCENTILE,
            percentile_low=1.0,
            percentile_high=99.0,
        ))
        ds = pydicom.dcmread("mammogram.dcm")
        processed = processor.process(ds)
        # processed: float32 array in [0, 1], MONOCHROME2
    """

    def __init__(
        self,
        config: Optional[PixelProcessingConfig] = None,
        manufacturer_profiles: Optional[Dict[str, ManufacturerProfile]] = None,
    ) -> None:
        self.config = config or PixelProcessingConfig()
        self.profiles = manufacturer_profiles or _MANUFACTURER_PROFILES

    def process(
        self,
        ds: Dataset,
        manufacturer_normalized: Optional[str] = None,
    ) -> np.ndarray:
        """Process pixel data from a DICOM dataset.

        Args:
            ds: pydicom Dataset with pixel data.
            manufacturer_normalized: Canonical manufacturer name for corrections.

        Returns:
            Processed pixel array as float32.

        Raises:
            ValueError: If pixel data cannot be extracted.
        """
        # Extract raw pixels
        try:
            pixel_array = ds.pixel_array.copy()
        except Exception as exc:
            raise ValueError(f"Cannot extract pixel data: {exc}") from exc

        logger.debug(
            "Input: shape=%s, dtype=%s, range=[%s, %s]",
            pixel_array.shape,
            pixel_array.dtype,
            pixel_array.min(),
            pixel_array.max(),
        )

        # Step 1: Manufacturer-specific pre-corrections
        if self.config.apply_manufacturer_corrections and manufacturer_normalized:
            pixel_array = self._apply_manufacturer_corrections(
                pixel_array, ds, manufacturer_normalized
            )

        # Step 2: Modality LUT
        if self.config.apply_modality_lut:
            pixel_array = apply_modality_lut(pixel_array, ds)

        # Step 3: VOI LUT
        if self.config.apply_voi_lut:
            pixel_array = apply_voi_lut(pixel_array, ds)

        # Step 4: Photometric interpretation
        photometric = str(getattr(ds, "PhotometricInterpretation", "MONOCHROME2"))
        pixel_array = handle_photometric_interpretation(
            pixel_array, photometric, self.config.target_photometric
        )

        # Step 5: Bit depth normalization
        bits_stored = int(getattr(ds, "BitsStored", 16) or 16)
        pixel_rep = int(getattr(ds, "PixelRepresentation", 0) or 0)
        pixel_array = normalize_bit_depth(pixel_array, bits_stored, pixel_rep, self.config)

        # Step 6: Convert to output dtype
        output = pixel_array.astype(getattr(np, self.config.output_dtype))

        logger.debug(
            "Output: shape=%s, dtype=%s, range=[%.4f, %.4f]",
            output.shape,
            output.dtype,
            float(output.min()),
            float(output.max()),
        )

        return output

    def _apply_manufacturer_corrections(
        self,
        pixel_array: np.ndarray,
        ds: Dataset,
        manufacturer: str,
    ) -> np.ndarray:
        """Apply manufacturer-specific pixel corrections.

        Args:
            pixel_array: Raw pixel array.
            ds: DICOM dataset.
            manufacturer: Canonical manufacturer name.

        Returns:
            Corrected pixel array.
        """
        profile = self.profiles.get(manufacturer)
        if profile is None:
            return pixel_array

        arr = pixel_array.astype(np.float64)

        # Apply offset
        if profile.pixel_offset != 0:
            arr += profile.pixel_offset

        # Apply log transform (some GE raw data)
        if profile.apply_log_transform:
            arr = np.clip(arr, 1.0, None)
            arr = np.log(arr) / np.log(profile.log_transform_base)

        # Mask dead columns
        if profile.dead_column_indices:
            for col_idx in profile.dead_column_indices:
                if 0 <= col_idx < arr.shape[-1]:
                    # Interpolate from neighbors
                    left = max(0, col_idx - 1)
                    right = min(arr.shape[-1] - 1, col_idx + 1)
                    arr[..., col_idx] = (arr[..., left] + arr[..., right]) / 2.0

        return arr
