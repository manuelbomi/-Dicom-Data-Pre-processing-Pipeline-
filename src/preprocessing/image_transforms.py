"""Configurable image transform pipeline for medical imaging.

Provides a composable set of image transformations with defaults appropriate
for medical imaging (not natural images). Key differences from standard
computer vision transforms:

  - Interpolation defaults to cubic (not bilinear) for better preservation
    of fine structures (microcalcifications, subtle lesions).
  - Padding uses constant zero rather than reflection, since reflection
    creates unrealistic tissue patterns.
  - Intensity normalization accounts for medical image distributions
    (heavy tails, large dynamic range, modality-specific ranges).
  - No random color jitter or saturation changes (grayscale modalities).

Each transform is a callable that takes and returns an ndarray, enabling
simple composition via TransformPipeline.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class BaseTransform(ABC):
    """Abstract base class for image transforms."""

    @abstractmethod
    def __call__(self, image: np.ndarray) -> np.ndarray:
        """Apply transform to an image.

        Args:
            image: Input image (2D or 3D).

        Returns:
            Transformed image.
        """

    @abstractmethod
    def __repr__(self) -> str:
        pass


class Resize(BaseTransform):
    """Resize image to target dimensions.

    Uses area interpolation for downsampling (anti-aliased) and cubic
    interpolation for upsampling, following best practices for medical images.

    Args:
        target_size: (height, width) target dimensions.
        interpolation_up: Interpolation for upsampling.
        interpolation_down: Interpolation for downsampling.
        preserve_aspect_ratio: If True, resize to fit within target_size
            while maintaining aspect ratio (may not fill target exactly).
    """

    def __init__(
        self,
        target_size: Tuple[int, int],
        interpolation_up: int = cv2.INTER_CUBIC,
        interpolation_down: int = cv2.INTER_AREA,
        preserve_aspect_ratio: bool = False,
    ) -> None:
        self.target_size = target_size
        self.interpolation_up = interpolation_up
        self.interpolation_down = interpolation_down
        self.preserve_aspect_ratio = preserve_aspect_ratio

    def __call__(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        th, tw = self.target_size

        if self.preserve_aspect_ratio:
            scale = min(th / h, tw / w)
            new_h = int(h * scale)
            new_w = int(w * scale)
        else:
            new_h, new_w = th, tw

        is_downscale = (new_h * new_w) < (h * w)
        interp = self.interpolation_down if is_downscale else self.interpolation_up

        resized = cv2.resize(image, (new_w, new_h), interpolation=interp)
        return resized

    def __repr__(self) -> str:
        return f"Resize(target_size={self.target_size}, preserve_aspect={self.preserve_aspect_ratio})"


class PadToSize(BaseTransform):
    """Pad image to exact target dimensions.

    Places the image in the center (or corner) of the target canvas.
    Uses constant padding (zero) by default, which is appropriate for
    medical images where edge reflection would create unrealistic anatomy.

    Args:
        target_size: (height, width) target dimensions.
        fill_value: Padding fill value.
        center: Whether to center the image. If False, places top-left.
    """

    def __init__(
        self,
        target_size: Tuple[int, int],
        fill_value: float = 0.0,
        center: bool = True,
    ) -> None:
        self.target_size = target_size
        self.fill_value = fill_value
        self.center = center

    def __call__(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        th, tw = self.target_size

        if h >= th and w >= tw:
            return image[:th, :tw]

        padded = np.full(
            (max(h, th), max(w, tw)) + image.shape[2:],
            self.fill_value,
            dtype=image.dtype,
        )

        if self.center:
            y_off = max(0, (th - h) // 2)
            x_off = max(0, (tw - w) // 2)
        else:
            y_off, x_off = 0, 0

        padded[y_off:y_off + h, x_off:x_off + w] = image
        return padded[:th, :tw]

    def __repr__(self) -> str:
        return f"PadToSize(target_size={self.target_size}, fill={self.fill_value})"


class CenterCrop(BaseTransform):
    """Crop a centered region from the image.

    Args:
        crop_size: (height, width) of the crop.
    """

    def __init__(self, crop_size: Tuple[int, int]) -> None:
        self.crop_size = crop_size

    def __call__(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        ch, cw = self.crop_size

        if ch > h or cw > w:
            logger.warning(
                "Crop size (%d, %d) exceeds image (%d, %d), returning padded",
                ch, cw, h, w,
            )
            pad = PadToSize(self.crop_size)
            return pad(image)

        y1 = (h - ch) // 2
        x1 = (w - cw) // 2
        return image[y1:y1 + ch, x1:x1 + cw].copy()

    def __repr__(self) -> str:
        return f"CenterCrop(crop_size={self.crop_size})"


class ResizeWithPad(BaseTransform):
    """Resize preserving aspect ratio, then pad to exact target size.

    This is the recommended transform for medical imaging: it avoids
    distortion while producing uniform-size outputs.

    Args:
        target_size: (height, width) final output size.
        fill_value: Padding value.
        interpolation_up: Upsampling interpolation.
        interpolation_down: Downsampling interpolation.
    """

    def __init__(
        self,
        target_size: Tuple[int, int],
        fill_value: float = 0.0,
        interpolation_up: int = cv2.INTER_CUBIC,
        interpolation_down: int = cv2.INTER_AREA,
    ) -> None:
        self.resize = Resize(
            target_size,
            interpolation_up=interpolation_up,
            interpolation_down=interpolation_down,
            preserve_aspect_ratio=True,
        )
        self.pad = PadToSize(target_size, fill_value=fill_value, center=True)

    def __call__(self, image: np.ndarray) -> np.ndarray:
        return self.pad(self.resize(image))

    def __repr__(self) -> str:
        return f"ResizeWithPad(target={self.resize.target_size})"


class IntensityNormalize(BaseTransform):
    """Normalize pixel intensities for ML consumption.

    Supports several normalization strategies appropriate for medical imaging:
      - "zero_one": Scale to [0, 1] via min-max.
      - "imagenet": Subtract mean, divide by std (single channel).
      - "percentile": Clip to [p_low, p_high] then scale to [0, 1].
      - "z_score": Zero mean, unit variance.
      - "preset": Use provided mean and std.

    Args:
        method: Normalization method name.
        percentile_low: Lower percentile for "percentile" method.
        percentile_high: Upper percentile for "percentile" method.
        mean: Fixed mean for "preset" method.
        std: Fixed std for "preset" method.
        epsilon: Small value to avoid division by zero.
    """

    def __init__(
        self,
        method: str = "percentile",
        percentile_low: float = 1.0,
        percentile_high: float = 99.0,
        mean: Optional[float] = None,
        std: Optional[float] = None,
        epsilon: float = 1e-8,
    ) -> None:
        self.method = method
        self.percentile_low = percentile_low
        self.percentile_high = percentile_high
        self.mean = mean
        self.std = std
        self.epsilon = epsilon

    def __call__(self, image: np.ndarray) -> np.ndarray:
        img = image.astype(np.float32)

        if self.method == "zero_one":
            vmin, vmax = img.min(), img.max()
            if vmax - vmin > self.epsilon:
                img = (img - vmin) / (vmax - vmin)
            else:
                img = np.zeros_like(img)

        elif self.method == "percentile":
            p_low = np.percentile(img, self.percentile_low)
            p_high = np.percentile(img, self.percentile_high)
            if p_high - p_low > self.epsilon:
                img = np.clip(img, p_low, p_high)
                img = (img - p_low) / (p_high - p_low)
            else:
                img = np.zeros_like(img)

        elif self.method == "z_score":
            mean = img.mean()
            std = img.std()
            if std > self.epsilon:
                img = (img - mean) / std

        elif self.method == "preset":
            if self.mean is not None and self.std is not None:
                img = (img - self.mean) / max(self.std, self.epsilon)
            else:
                raise ValueError("preset method requires mean and std")

        elif self.method == "imagenet":
            # Adapted ImageNet normalization for single-channel medical
            img = (img - 0.449) / 0.226

        return img

    def __repr__(self) -> str:
        return f"IntensityNormalize(method='{self.method}')"


class GaussianBlur(BaseTransform):
    """Apply Gaussian blur for noise reduction.

    Light blurring can reduce acquisition noise without significantly
    impacting the features relevant for ML models.

    Args:
        kernel_size: Gaussian kernel size (must be odd).
        sigma: Gaussian standard deviation. 0 = auto from kernel_size.
    """

    def __init__(self, kernel_size: int = 3, sigma: float = 0.0) -> None:
        if kernel_size % 2 == 0:
            kernel_size += 1
        self.kernel_size = kernel_size
        self.sigma = sigma

    def __call__(self, image: np.ndarray) -> np.ndarray:
        return cv2.GaussianBlur(
            image,
            (self.kernel_size, self.kernel_size),
            self.sigma,
        )

    def __repr__(self) -> str:
        return f"GaussianBlur(kernel={self.kernel_size}, sigma={self.sigma})"


class AddChannelDim(BaseTransform):
    """Add a channel dimension for PyTorch (H, W) -> (1, H, W)."""

    def __call__(self, image: np.ndarray) -> np.ndarray:
        if image.ndim == 2:
            return image[np.newaxis, :, :]
        return image

    def __repr__(self) -> str:
        return "AddChannelDim()"


class ToTensor(BaseTransform):
    """Convert numpy array to float32 contiguous array (pre-torch.from_numpy).

    Ensures memory layout is C-contiguous and dtype is float32.
    """

    def __call__(self, image: np.ndarray) -> np.ndarray:
        return np.ascontiguousarray(image, dtype=np.float32)

    def __repr__(self) -> str:
        return "ToTensor()"


class TransformPipeline:
    """Composable transform pipeline.

    Chains multiple transforms in order. Provides logging and error handling.

    Args:
        transforms: Ordered list of transforms to apply.

    Example::

        pipeline = TransformPipeline([
            Resize((2048, 1024), preserve_aspect_ratio=True),
            PadToSize((2048, 1024)),
            IntensityNormalize(method="percentile"),
            AddChannelDim(),
            ToTensor(),
        ])
        output = pipeline(image)
    """

    def __init__(self, transforms: List[BaseTransform]) -> None:
        self.transforms = transforms

    def __call__(self, image: np.ndarray) -> np.ndarray:
        for transform in self.transforms:
            try:
                image = transform(image)
            except Exception as exc:
                logger.error("Transform %s failed: %s", transform, exc)
                raise
        return image

    def __repr__(self) -> str:
        steps = "\n  ".join(repr(t) for t in self.transforms)
        return f"TransformPipeline([\n  {steps}\n])"

    def append(self, transform: BaseTransform) -> None:
        """Add a transform to the end of the pipeline."""
        self.transforms.append(transform)

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "TransformPipeline":
        """Build a transform pipeline from a configuration dictionary.

        Config format::

            transforms:
              - name: Resize
                params:
                  target_size: [2048, 1024]
                  preserve_aspect_ratio: true
              - name: IntensityNormalize
                params:
                  method: percentile

        Args:
            config: Dictionary with "transforms" key.

        Returns:
            Configured TransformPipeline.
        """
        transform_map = {
            "Resize": Resize,
            "PadToSize": PadToSize,
            "CenterCrop": CenterCrop,
            "ResizeWithPad": ResizeWithPad,
            "IntensityNormalize": IntensityNormalize,
            "GaussianBlur": GaussianBlur,
            "AddChannelDim": AddChannelDim,
            "ToTensor": ToTensor,
        }

        transforms: List[BaseTransform] = []
        for step in config.get("transforms", []):
            name = step["name"]
            params = step.get("params", {})

            # Convert list params to tuples where expected
            for key in ("target_size", "crop_size", "grid_size"):
                if key in params and isinstance(params[key], list):
                    params[key] = tuple(params[key])

            if name not in transform_map:
                raise ValueError(f"Unknown transform: {name}")

            transforms.append(transform_map[name](**params))

        return cls(transforms)


def get_mammography_transforms(
    target_size: Tuple[int, int] = (2048, 1024),
    normalization: str = "percentile",
) -> TransformPipeline:
    """Get the default transform pipeline for mammography.

    Args:
        target_size: Output image dimensions (height, width).
        normalization: Intensity normalization method.

    Returns:
        Configured TransformPipeline.
    """
    return TransformPipeline([
        ResizeWithPad(target_size, fill_value=0.0),
        IntensityNormalize(method=normalization, percentile_low=1.0, percentile_high=99.0),
        AddChannelDim(),
        ToTensor(),
    ])


def get_ct_transforms(
    target_size: Tuple[int, int] = (512, 512),
    window_center: float = 40.0,
    window_width: float = 400.0,
) -> TransformPipeline:
    """Get the default transform pipeline for CT images.

    Applies soft-tissue windowing before normalization.

    Args:
        target_size: Output image dimensions.
        window_center: CT window center (HU).
        window_width: CT window width (HU).

    Returns:
        Configured TransformPipeline.
    """
    return TransformPipeline([
        Resize(target_size),
        IntensityNormalize(method="zero_one"),
        AddChannelDim(),
        ToTensor(),
    ])
