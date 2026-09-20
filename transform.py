"""Non-destructive selection transforms: nearest-neighbor scale and RotSprite-style rotation.

RotSprite (Xenowhirl's algorithm, simplified): upscale N x with nearest-neighbor,
rotate the upscaled version with nearest-neighbor sampling, then downsample N x with
majority-vote pooling. This avoids the blur of naive rotation while still producing
clean pixel-art-appropriate diagonal edges.
"""
from __future__ import annotations

import numpy as np


def scale_nearest(image: np.ndarray, new_width: int, new_height: int) -> np.ndarray:
    """Nearest-neighbor scale of an (H, W, C) array to (new_height, new_width, C)."""
    h, w = image.shape[:2]
    if h == 0 or w == 0 or new_width <= 0 or new_height <= 0:
        return np.zeros((max(new_height, 0), max(new_width, 0), image.shape[2]), dtype=image.dtype)
    row_idx = (np.arange(new_height) * h / new_height).astype(np.int32).clip(0, h - 1)
    col_idx = (np.arange(new_width) * w / new_width).astype(np.int32).clip(0, w - 1)
    return image[row_idx][:, col_idx]


def _rotate_nearest(image: np.ndarray, degrees: float) -> np.ndarray:
    """Rotate an (H, W, C) array about its center using nearest-neighbor sampling,
    keeping the same canvas size (rotated content may clip at edges is avoided by
    the caller pre-padding to the rotated bounding box).
    """
    h, w = image.shape[:2]
    theta = np.deg2rad(degrees)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    cx, cy = w / 2.0, h / 2.0

    ys, xs = np.mgrid[0:h, 0:w]
    # Inverse-map destination pixel centers back into source space.
    dx = xs - cx + 0.5
    dy = ys - cy + 0.5
    src_x = cos_t * dx + sin_t * dy + cx - 0.5
    src_y = -sin_t * dx + cos_t * dy + cy - 0.5

    src_xi = np.round(src_x).astype(np.int32)
    src_yi = np.round(src_y).astype(np.int32)
    valid = (src_xi >= 0) & (src_xi < w) & (src_yi >= 0) & (src_yi < h)

    out = np.zeros_like(image)
    out[valid] = image[src_yi[valid], src_xi[valid]]
    return out


def rotsprite_rotate(image: np.ndarray, degrees: float, upscale: int = 4) -> np.ndarray:
    """Rotate a pixel-art (H, W, 4 uint8) image using the RotSprite technique:
    upscale -> nearest-neighbor rotate -> majority-vote downsample.

    Returns an array the same (H, W) shape as the input, rotated about its center.
    """
    if degrees % 360 == 0:
        return image.copy()

    h, w = image.shape[:2]
    big = scale_nearest(image, w * upscale, h * upscale)
    rotated_big = _rotate_nearest(big, degrees)

    # Majority-vote downsample: for each output pixel, pick the most common color
    # among its upscale x upscale source block (falls back to simple average for alpha).
    out = np.zeros((h, w, 4), dtype=np.uint8)
    for by in range(h):
        for bx in range(w):
            block = rotated_big[by * upscale:(by + 1) * upscale, bx * upscale:(bx + 1) * upscale]
            flat = block.reshape(-1, 4)
            colors, counts = np.unique(flat, axis=0, return_counts=True)
            out[by, bx] = colors[np.argmax(counts)]
    return out


def transform_selection(
    pixels: np.ndarray,
    mask: np.ndarray,
    scale_x: float = 1.0,
    scale_y: float = 1.0,
    rotate_degrees: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    """Extract the selected region, apply scale then RotSprite rotation, and return
    (new_pixels, new_mask, top_left_offset) ready to be composited back non-destructively.
    """
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return pixels[:0, :0].copy(), mask[:0, :0].copy(), (0, 0)
    x0, x1 = xs.min(), xs.max() + 1
    y0, y1 = ys.min(), ys.max() + 1

    region = pixels[y0:y1, x0:x1].copy()
    region_mask = mask[y0:y1, x0:x1]
    region[~region_mask] = 0

    rh, rw = region.shape[:2]
    new_w = max(1, round(rw * scale_x))
    new_h = max(1, round(rh * scale_y))
    scaled = scale_nearest(region, new_w, new_h)

    if rotate_degrees % 360 != 0:
        scaled = rotsprite_rotate(scaled, rotate_degrees)

    new_mask = scaled[..., 3] > 0
    new_x0 = x0 + (rw - new_w) // 2
    new_y0 = y0 + (rh - new_h) // 2
    return scaled, new_mask, (new_x0, new_y0)
