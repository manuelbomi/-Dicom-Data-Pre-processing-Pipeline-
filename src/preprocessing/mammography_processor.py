"""Mammography-specific image processing.

Implements the specialized preprocessing steps required for screening and
diagnostic mammography images before ML model consumption:

  - **Laterality normalization**: Flip all images to a consistent orientation
    (e.g., all left-facing) to reduce spatial bias in models.
  - **Implant detection**: Identify breast implants from metadata and pixel
    patterns to flag or mask them during training.
  - **Pectoral muscle segmentation**: Detect and optionally mask the pectoral
    muscle region in MLO views, which is irrelevant for parenchymal analysis.
  - **Breast boundary detection**: Segment the breast region from the
    background to compute ROI masks and remove scanner artifacts.
  - **CLAHE enhancement**: Apply Contrast-Limited Adaptive Histogram
    Equalization for improved tissue visualization.

These operations are critical for mammography AI models (e.g., breast cancer
detection, density estimation) and follow established practices from the
literature (Ribli et al. 2018, Wu et al. 2020, McKinney et al. 2020).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class Orientation(Enum):
    """Breast orientation in the image."""

    LEFT_FACING = "left"    # Breast tissue on the left side
    RIGHT_FACING = "right"  # Breast tissue on the right side
    UNKNOWN = "unknown"


class ViewPosition(Enum):
    """Standard mammographic view positions."""

    CC = "CC"     # Craniocaudal
    MLO = "MLO"   # Mediolateral oblique
    ML = "ML"     # Mediolateral
    LM = "LM"     # Lateromedial
    OTHER = "OTHER"


@dataclass
class MammographyConfig:
    """Configuration for mammography preprocessing."""

    # Laterality
    target_orientation: Orientation = Orientation.LEFT_FACING
    auto_detect_orientation: bool = True

    # Breast segmentation
    breast_threshold_method: str = "otsu"  # "otsu", "percentile", "fixed"
    breast_threshold_percentile: float = 5.0
    breast_min_area_fraction: float = 0.05  # Min breast area as fraction of image
    morphology_kernel_size: int = 25
    remove_artifacts: bool = True

    # Pectoral muscle
    pectoral_removal_enabled: bool = True
    pectoral_margin_pixels: int = 10
    pectoral_min_angle_deg: float = 20.0
    pectoral_max_angle_deg: float = 75.0

    # CLAHE
    clahe_enabled: bool = True
    clahe_clip_limit: float = 2.0
    clahe_grid_size: Tuple[int, int] = (8, 8)

    # Implant
    implant_detection_enabled: bool = True
    implant_intensity_threshold: float = 0.95  # Very bright = implant

    # Output
    crop_to_breast: bool = True
    padding_pixels: int = 10


@dataclass
class MammogramAnalysis:
    """Analysis results for a single mammogram."""

    orientation: Orientation = Orientation.UNKNOWN
    was_flipped: bool = False
    breast_mask: Optional[np.ndarray] = None
    breast_bbox: Optional[Tuple[int, int, int, int]] = None  # (y1, x1, y2, x2)
    breast_area_pixels: int = 0
    breast_area_fraction: float = 0.0
    pectoral_mask: Optional[np.ndarray] = None
    pectoral_area_fraction: float = 0.0
    implant_detected: bool = False
    implant_mask: Optional[np.ndarray] = None
    view_position: ViewPosition = ViewPosition.OTHER


def detect_orientation(image: np.ndarray) -> Orientation:
    """Detect breast orientation by analyzing pixel mass distribution.

    The breast appears as a bright region against a dark background.
    We compare the summed intensity of the left half vs. right half
    to determine which side the breast is on.

    Args:
        image: 2D grayscale image (float32 or uint).

    Returns:
        Detected Orientation.
    """
    if image.ndim != 2:
        raise ValueError(f"Expected 2D image, got shape {image.shape}")

    h, w = image.shape
    mid = w // 2

    left_mass = float(image[:, :mid].sum())
    right_mass = float(image[:, mid:].sum())

    # Use a margin to avoid ambiguous cases
    ratio = left_mass / max(right_mass, 1e-8)
    if ratio > 1.1:
        return Orientation.LEFT_FACING
    elif ratio < 0.9:
        return Orientation.RIGHT_FACING
    else:
        # Ambiguous -- use column-wise center of mass
        col_sums = image.sum(axis=0).astype(np.float64)
        total = col_sums.sum()
        if total < 1e-8:
            return Orientation.UNKNOWN
        com = float(np.arange(w).dot(col_sums) / total)
        return Orientation.LEFT_FACING if com < mid else Orientation.RIGHT_FACING


def normalize_laterality(
    image: np.ndarray,
    laterality: Optional[str] = None,
    target: Orientation = Orientation.LEFT_FACING,
    auto_detect: bool = True,
) -> Tuple[np.ndarray, bool]:
    """Normalize breast laterality by flipping if necessary.

    All images are oriented so the breast faces the target direction.
    This removes spatial bias that could confuse ML models.

    Args:
        image: 2D grayscale image.
        laterality: DICOM ImageLaterality ("L" or "R"), or None.
        target: Desired output orientation.
        auto_detect: Fall back to pixel-based detection if laterality unknown.

    Returns:
        Tuple of (normalized image, whether flip was applied).
    """
    # Determine current orientation
    if laterality == "R":
        current = Orientation.RIGHT_FACING
    elif laterality == "L":
        current = Orientation.LEFT_FACING
    elif auto_detect:
        current = detect_orientation(image)
    else:
        return image, False

    need_flip = current != target and current != Orientation.UNKNOWN

    if need_flip:
        image = np.fliplr(image).copy()
        logger.debug("Flipped image: %s -> %s", current.value, target.value)

    return image, need_flip


def segment_breast(
    image: np.ndarray,
    config: Optional[MammographyConfig] = None,
) -> np.ndarray:
    """Segment the breast region from background.

    Uses thresholding + morphological operations to create a binary mask
    of the breast tissue region. Handles the common artifacts:
    - Scanner borders and labels
    - Bright annotation markers
    - Partial collimation

    Args:
        image: 2D float32 image in [0, 1].
        config: Processing configuration.

    Returns:
        Binary mask (uint8, 0 or 255) of the breast region.
    """
    if config is None:
        config = MammographyConfig()

    h, w = image.shape

    # Convert to uint8 for OpenCV
    img_u8 = (image * 255).clip(0, 255).astype(np.uint8)

    # Threshold
    if config.breast_threshold_method == "otsu":
        _, mask = cv2.threshold(img_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif config.breast_threshold_method == "percentile":
        thresh = np.percentile(img_u8[img_u8 > 0], config.breast_threshold_percentile)
        _, mask = cv2.threshold(img_u8, int(thresh), 255, cv2.THRESH_BINARY)
    else:
        _, mask = cv2.threshold(img_u8, 10, 255, cv2.THRESH_BINARY)

    # Morphological cleanup
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (config.morphology_kernel_size, config.morphology_kernel_size),
    )
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Keep only the largest connected component (the breast)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        logger.warning("No breast contour found")
        return mask

    min_area = h * w * config.breast_min_area_fraction
    largest = max(contours, key=cv2.contourArea)

    if cv2.contourArea(largest) < min_area:
        logger.warning(
            "Largest contour area (%d) below threshold (%d)",
            cv2.contourArea(largest),
            min_area,
        )

    # Create clean mask from largest contour
    clean_mask = np.zeros_like(mask)
    cv2.drawContours(clean_mask, [largest], -1, 255, thickness=cv2.FILLED)

    # Remove artifacts (small bright regions not connected to breast)
    if config.remove_artifacts:
        for cnt in contours:
            if cv2.contourArea(cnt) < min_area * 0.1:
                cv2.drawContours(clean_mask, [cnt], -1, 0, thickness=cv2.FILLED)

    return clean_mask


def segment_pectoral_muscle(
    image: np.ndarray,
    breast_mask: np.ndarray,
    orientation: Orientation = Orientation.LEFT_FACING,
    config: Optional[MammographyConfig] = None,
) -> np.ndarray:
    """Segment the pectoral muscle in MLO view mammograms.

    The pectoral muscle appears as a bright triangular region in the
    upper-left (for left-facing) or upper-right corner of MLO views.
    We use a combination of thresholding and line fitting.

    Args:
        image: 2D float32 image in [0, 1].
        breast_mask: Binary breast mask.
        orientation: Current image orientation (after laterality normalization).
        config: Processing configuration.

    Returns:
        Binary mask (uint8) of the pectoral muscle region.
    """
    if config is None:
        config = MammographyConfig()

    h, w = image.shape
    pec_mask = np.zeros((h, w), dtype=np.uint8)

    # The pectoral muscle is in the upper corner on the chest wall side
    # For left-facing images, the chest wall is on the left
    if orientation == Orientation.LEFT_FACING:
        roi = image[:h // 2, :w // 3]
        x_offset, y_offset = 0, 0
    else:
        roi = image[:h // 2, 2 * w // 3:]
        x_offset, y_offset = 2 * w // 3, 0

    if roi.size == 0:
        return pec_mask

    # Threshold the ROI to find the bright pectoral region
    roi_u8 = (roi * 255).clip(0, 255).astype(np.uint8)
    thresh = np.percentile(roi_u8[roi_u8 > 0], 80) if (roi_u8 > 0).any() else 128
    _, binary = cv2.threshold(roi_u8, int(thresh), 255, cv2.THRESH_BINARY)

    # Apply mask intersection with breast
    breast_roi = breast_mask[y_offset:y_offset + roi.shape[0],
                              x_offset:x_offset + roi.shape[1]]
    binary = cv2.bitwise_and(binary, breast_roi)

    # Edge detection for pectoral muscle boundary
    edges = cv2.Canny(binary, 50, 150)

    # Hough line detection to find the pectoral muscle boundary
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=min(roi.shape) // 4)

    if lines is not None:
        # Find the best line within expected angle range
        best_line = None
        best_score = -1

        for line in lines:
            rho, theta = line[0]
            angle_deg = np.degrees(theta)

            # Pectoral boundary angle constraints
            if (config.pectoral_min_angle_deg <= angle_deg <= config.pectoral_max_angle_deg):
                score = abs(rho)  # Prefer lines further from origin
                if score > best_score:
                    best_score = score
                    best_line = (rho, theta)

        if best_line is not None:
            rho, theta = best_line
            # Create mask from the line: fill the triangle above/left of line
            cos_t = np.cos(theta)
            sin_t = np.sin(theta)

            for y in range(roi.shape[0]):
                for x in range(roi.shape[1]):
                    if x * cos_t + y * sin_t < rho + config.pectoral_margin_pixels:
                        pec_mask[y + y_offset, x + x_offset] = 255
    else:
        # Fallback: use intensity-based detection
        high_intensity = roi_u8 > int(thresh)
        pec_mask[y_offset:y_offset + roi.shape[0],
                 x_offset:x_offset + roi.shape[1]] = (high_intensity * 255).astype(np.uint8)

    return pec_mask


def detect_implant(
    image: np.ndarray,
    breast_mask: np.ndarray,
    config: Optional[MammographyConfig] = None,
    metadata_implant: Optional[bool] = None,
) -> Tuple[bool, Optional[np.ndarray]]:
    """Detect breast implant from pixel data and/or metadata.

    Implants appear as very bright, smooth, large regions. We detect them
    via intensity thresholding within the breast mask. Metadata
    (BreastImplantPresent tag) is used as a prior when available.

    Args:
        image: 2D float32 image in [0, 1].
        breast_mask: Binary breast mask.
        config: Processing configuration.
        metadata_implant: Value of BreastImplantPresent tag, if available.

    Returns:
        Tuple of (implant_detected, implant_mask).
    """
    if config is None:
        config = MammographyConfig()

    # If metadata confirms implant, trust it
    if metadata_implant is True:
        # Still compute the mask for downstream use
        pass

    # Find very bright regions within the breast
    masked_image = image * (breast_mask > 0).astype(np.float32)
    breast_pixels = masked_image[breast_mask > 0]

    if len(breast_pixels) == 0:
        return False, None

    # Implants have very high intensity with low variance (smooth silicone)
    threshold = config.implant_intensity_threshold
    bright_mask = (masked_image > threshold).astype(np.uint8) * 255

    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    bright_mask = cv2.morphologyEx(bright_mask, cv2.MORPH_CLOSE, kernel)
    bright_mask = cv2.morphologyEx(bright_mask, cv2.MORPH_OPEN, kernel)

    # Check if bright region is large enough to be an implant
    bright_area = np.count_nonzero(bright_mask)
    breast_area = np.count_nonzero(breast_mask)

    if breast_area == 0:
        return False, None

    bright_fraction = bright_area / breast_area

    # Implants typically occupy 15-50% of the breast region
    implant_detected = bright_fraction > 0.10

    if metadata_implant is True:
        implant_detected = True

    if implant_detected:
        logger.info(
            "Implant detected: bright fraction=%.2f%% of breast",
            bright_fraction * 100,
        )

    return implant_detected, bright_mask if implant_detected else None


def apply_clahe(
    image: np.ndarray,
    clip_limit: float = 2.0,
    grid_size: Tuple[int, int] = (8, 8),
) -> np.ndarray:
    """Apply Contrast-Limited Adaptive Histogram Equalization.

    CLAHE enhances local contrast while preventing noise amplification,
    making it ideal for mammography where tissue density variations
    span multiple scales. It operates on tiles independently, which
    preserves both subtle microcalcifications and gross architectural
    distortions.

    Args:
        image: 2D float32 image in [0, 1].
        clip_limit: Contrast limit for histogram clipping.
        grid_size: Tile grid dimensions.

    Returns:
        CLAHE-enhanced float32 image in [0, 1].
    """
    img_u8 = (image * 255).clip(0, 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    enhanced = clahe.apply(img_u8)
    return enhanced.astype(np.float32) / 255.0


def crop_to_breast_region(
    image: np.ndarray,
    breast_mask: np.ndarray,
    padding: int = 10,
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """Crop image to the bounding box of the breast region.

    Args:
        image: 2D image.
        breast_mask: Binary breast mask.
        padding: Extra pixels around the bounding box.

    Returns:
        Tuple of (cropped image, bounding box as (y1, x1, y2, x2)).
    """
    coords = np.where(breast_mask > 0)
    if len(coords[0]) == 0:
        return image, (0, 0, image.shape[0], image.shape[1])

    y1 = max(0, int(coords[0].min()) - padding)
    y2 = min(image.shape[0], int(coords[0].max()) + padding + 1)
    x1 = max(0, int(coords[1].min()) - padding)
    x2 = min(image.shape[1], int(coords[1].max()) + padding + 1)

    return image[y1:y2, x1:x2].copy(), (y1, x1, y2, x2)


class MammographyProcessor:
    """Complete mammography image preprocessing pipeline.

    Orchestrates all mammography-specific processing steps in the correct
    order, producing a normalized, cleaned image with analysis metadata.

    Args:
        config: Mammography processing configuration.

    Example::

        processor = MammographyProcessor(MammographyConfig(
            clahe_enabled=True,
            pectoral_removal_enabled=True,
            crop_to_breast=True,
        ))
        result = processor.process(
            image=pixel_array,
            laterality="L",
            view_position="MLO",
            implant_present=None,
        )
        processed_image = result["image"]
        analysis = result["analysis"]
    """

    def __init__(self, config: Optional[MammographyConfig] = None) -> None:
        self.config = config or MammographyConfig()

    def process(
        self,
        image: np.ndarray,
        laterality: Optional[str] = None,
        view_position: Optional[str] = None,
        implant_present: Optional[bool] = None,
    ) -> Dict[str, Union[np.ndarray, MammogramAnalysis]]:
        """Process a single mammogram image.

        Args:
            image: 2D float32 image in [0, 1].
            laterality: DICOM ImageLaterality ("L" or "R").
            view_position: DICOM ViewPosition ("CC", "MLO", etc.).
            implant_present: From DICOM BreastImplantPresent tag.

        Returns:
            Dict with "image" (processed array) and "analysis" (MammogramAnalysis).
        """
        analysis = MammogramAnalysis()
        if view_position:
            try:
                analysis.view_position = ViewPosition(view_position)
            except ValueError:
                analysis.view_position = ViewPosition.OTHER

        # Ensure 2D
        if image.ndim == 3 and image.shape[-1] == 1:
            image = image[:, :, 0]
        if image.ndim != 2:
            raise ValueError(f"Expected 2D mammogram, got shape {image.shape}")

        # Step 1: Laterality normalization
        image, was_flipped = normalize_laterality(
            image,
            laterality=laterality,
            target=self.config.target_orientation,
            auto_detect=self.config.auto_detect_orientation,
        )
        analysis.was_flipped = was_flipped
        analysis.orientation = self.config.target_orientation

        # Step 2: Breast segmentation
        breast_mask = segment_breast(image, self.config)
        analysis.breast_mask = breast_mask
        analysis.breast_area_pixels = int(np.count_nonzero(breast_mask))
        total_pixels = image.shape[0] * image.shape[1]
        analysis.breast_area_fraction = analysis.breast_area_pixels / max(total_pixels, 1)

        # Step 3: Implant detection
        if self.config.implant_detection_enabled:
            implant_detected, implant_mask = detect_implant(
                image, breast_mask, self.config, metadata_implant=implant_present
            )
            analysis.implant_detected = implant_detected
            analysis.implant_mask = implant_mask

        # Step 4: Pectoral muscle segmentation (MLO views only)
        if (
            self.config.pectoral_removal_enabled
            and analysis.view_position == ViewPosition.MLO
        ):
            pec_mask = segment_pectoral_muscle(
                image, breast_mask, analysis.orientation, self.config
            )
            analysis.pectoral_mask = pec_mask
            pec_area = int(np.count_nonzero(pec_mask))
            analysis.pectoral_area_fraction = pec_area / max(total_pixels, 1)

            # Mask out pectoral muscle
            image = image * (1 - pec_mask.astype(np.float32) / 255.0)

        # Step 5: Apply breast mask (zero out background)
        image = image * (breast_mask.astype(np.float32) / 255.0)

        # Step 6: CLAHE enhancement
        if self.config.clahe_enabled:
            image = apply_clahe(
                image,
                clip_limit=self.config.clahe_clip_limit,
                grid_size=self.config.clahe_grid_size,
            )
            # Re-apply mask after CLAHE
            image = image * (breast_mask.astype(np.float32) / 255.0)

        # Step 7: Crop to breast region
        if self.config.crop_to_breast:
            image, bbox = crop_to_breast_region(
                image, breast_mask, self.config.padding_pixels
            )
            analysis.breast_bbox = bbox

        return {"image": image, "analysis": analysis}
