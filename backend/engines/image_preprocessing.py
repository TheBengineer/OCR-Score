"""Optional image preprocessing for OCR engines.

Provides deskew (rotation correction) and binarization (OTSU thresholding)
using OpenCV.  All functions gracefully fall back to the original image
when OpenCV is not installed.

Typical usage::

    from backend.engines.image_preprocessing import preprocess_pipeline

    processed = preprocess_pipeline(image, {"max_angle": 10.0, "method": "otsu"})
"""

from __future__ import annotations

import logging
from typing import Any

from PIL import Image

logger = logging.getLogger(__name__)

# ── Optional OpenCV support ───────────────────────────────────────────────────

try:
    import cv2
    import numpy as np

    HAS_CV2 = True
except ImportError:
    cv2 = None  # type: ignore[assignment]
    np = None  # type: ignore[assignment]
    HAS_CV2 = False
    logger.warning("OpenCV (cv2) not available — image preprocessing disabled.")


# ── Public API ────────────────────────────────────────────────────────────────


def deskew(image: Image.Image, max_angle: float = 5.0) -> Image.Image:
    """Detect and correct image skew using OpenCV's ``minAreaRect``.

    The function converts the image to grayscale, binarises it with OTSU,
    finds the minimum-area rotated rectangle covering all non-zero pixels,
    and rotates the **original** image back to horizontal if the detected
    skew exceeds 0.5°.

    Args:
        image: Input PIL Image (RGBA, RGB, or grayscale).
        max_angle: Maximum absolute skew angle to correct (degrees).
                   Skews larger than this are left untouched to avoid
                   over-correcting intentionally rotated pages.

    Returns:
        Deskewed PIL Image, or the original image if:

        - OpenCV is not installed.
        - The detected skew angle is below 0.5°.
        - The detected skew angle exceeds *max_angle*.
        - The image could not be processed (e.g. non-standard input).
    """
    if not HAS_CV2:
        return image

    try:
        img_array = np.array(image)  # type: ignore[misc]
    except Exception:
        return image

    if img_array.dtype not in (np.uint8, np.uint16):
        return image

    gray = (
        cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        if len(img_array.shape) == 3
        else img_array
    )

    _, binary = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # Find all non-zero (text) points
    coords = cv2.findNonZero(binary)

    if coords is None or len(coords) < 10:
        # Too few points to determine skew
        return image

    # Compute minimum-area rectangle to derive skew angle
    angle = cv2.minAreaRect(coords)[-1]

    # Normalise angle to [-45, 45]
    if angle < -45:
        angle = 90 + angle
    elif angle > 45:
        angle = angle - 90

    if abs(angle) > max_angle:
        # Skew too large — likely an intentionally rotated page
        return image

    if abs(angle) <= 0.5:
        # Negligible skew — skip transform
        return image

    # Rotate the original (colour) image
    h, w = img_array.shape[:2]
    center = (w / 2.0, h / 2.0)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        img_array,
        rotation_matrix,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )

    return Image.fromarray(rotated)


def binarize(image: Image.Image, method: str = "otsu") -> Image.Image:
    """Convert image to grayscale and apply binary thresholding.

    Args:
        image: Input PIL Image (any mode).
        method: Thresholding method.  Currently only ``"otsu"`` is supported.

    Returns:
        Binary (mode ``L``) PIL Image, or the original image if OpenCV is
        not installed or the image could not be processed.

    Raises:
        ValueError: If *method* is not ``"otsu"``.
    """
    if not HAS_CV2:
        return image

    if method != "otsu":
        raise ValueError(f"Unsupported binarization method: {method!r}")

    try:
        img_array = np.array(image)  # type: ignore[misc]
    except Exception:
        return image

    if img_array.dtype not in (np.uint8, np.uint16):
        return image

    gray = (
        cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        if len(img_array.shape) == 3
        else img_array
    )

    _, binary = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    return Image.fromarray(binary)


def preprocess_pipeline(
    image: Image.Image,
    config: dict[str, Any] | None = None,
) -> Image.Image:
    """Run the full preprocessing pipeline: deskew → binarize.

    This is the high-level entry point called by engine implementations.
    Both steps are skipped if OpenCV is not installed.

    Args:
        image: Input PIL Image.
        config: Optional dictionary with overrides:

            - ``max_angle`` (float): Maximum deskew angle (default: ``5.0``).
            - ``method`` (str): Binarization method (default: ``"otsu"``).

    Returns:
        Preprocessed PIL Image, or the original if OpenCV is unavailable.
    """
    cfg = config or {}

    if not HAS_CV2:
        return image

    # RGBA → RGB — OpenCV does not handle alpha transparently
    if image.mode == "RGBA":
        image = image.convert("RGB")

    image = deskew(image, max_angle=float(cfg.get("max_angle", 5.0)))
    image = binarize(image, method=str(cfg.get("method", "otsu")))

    return image
