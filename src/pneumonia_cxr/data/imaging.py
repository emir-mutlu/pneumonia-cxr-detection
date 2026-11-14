"""Image normalisation shared by data preparation and inference (no torch dependency).

Every image, whatever its source format (8-bit JPEG, 16-bit PNG, DICOM), goes through
the same two steps so that training and serving see identical pixels:

1. :func:`to_uint8_gray` - single channel, 0-255, MONOCHROME2 (bone is bright).
2. :func:`resize_square` - bilinear (anti-aliased) resize to ``cache_size`` x ``cache_size``.

The model input is then a further resize to ``img_size`` and scaling to [0, 1].
"""

from __future__ import annotations

import numpy as np
from PIL import Image

CACHE_SIZE = 256


def to_uint8_gray(img: Image.Image | np.ndarray, invert: bool = False) -> Image.Image:
    """Convert a PIL image or array of any bit depth to an 8-bit grayscale PIL image."""
    if isinstance(img, Image.Image):
        if img.mode in ("L", "RGB", "RGBA", "P", "LA", "CMYK", "YCbCr"):
            out = img.convert("L")
            arr = np.asarray(out)
        else:  # 16-bit / 32-bit / float modes
            arr = np.asarray(img)
    else:
        arr = np.asarray(img)

    if arr.ndim == 3:  # H x W x C array
        arr = arr[..., :3].mean(axis=-1)
    if arr.dtype != np.uint8:
        arr = arr.astype(np.float32)
        lo, hi = float(arr.min()), float(arr.max())
        arr = (arr - lo) / (hi - lo) * 255.0 if hi > lo else np.zeros_like(arr)
        arr = np.clip(np.rint(arr), 0, 255).astype(np.uint8)
    if invert:
        arr = 255 - arr
    return Image.fromarray(arr, mode="L")


def resize_square(img: Image.Image, size: int = CACHE_SIZE) -> np.ndarray:
    """Anti-aliased bilinear resize to ``size`` x ``size``; returns a uint8 array."""
    if img.mode != "L":
        img = img.convert("L")
    return np.asarray(img.resize((size, size), Image.Resampling.BILINEAR), dtype=np.uint8)


def preprocess_for_model(
    img: Image.Image | np.ndarray, img_size: int = 224, cache_size: int = CACHE_SIZE
) -> np.ndarray:
    """Full serving pipeline: any image -> float32 array of shape (1, 1, img_size, img_size)."""
    gray = to_uint8_gray(img)
    cached = Image.fromarray(resize_square(gray, cache_size), mode="L")
    arr = np.asarray(cached.resize((img_size, img_size), Image.Resampling.BILINEAR))
    return (arr.astype(np.float32) / 255.0)[None, None]
