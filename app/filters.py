"""Pixel-aware filter engine: palette quantization, outlines, ramp shading,
auto double-pixel cleanup, and drop shadow.
"""
from __future__ import annotations

import numpy as np

from palette import Palette, quantize_to_palette


def apply_palette_quantize(image: np.ndarray, palette: Palette) -> np.ndarray:
    """Map an RGBA image to the nearest colors in `palette`, preserving alpha=0 pixels."""
    result = quantize_to_palette(image, palette)
    result[..., 3] = image[..., 3]
    result[image[..., 3] == 0] = 0
    return result


def generate_outline(
    image: np.ndarray,
    color: tuple[int, int, int, int] = (0, 0, 0, 255),
    dithered: bool = False,
) -> np.ndarray:
    """Return a new RGBA image containing a 1px outline around the silhouette of
    `image` (where alpha > 0), drawn only into previously-transparent pixels.
    `image` is not modified; composite the result underneath/over the original as needed.
    """
    h, w = image.shape[:2]
    alpha = image[..., 3] > 0
    outline_mask = np.zeros((h, w), dtype=bool)

    shifts = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for dx, dy in shifts:
        shifted = np.zeros((h, w), dtype=bool)
        src_x0, src_x1 = max(0, -dx), w - max(0, dx)
        dst_x0, dst_x1 = max(0, dx), w - max(0, -dx)
        src_y0, src_y1 = max(0, -dy), h - max(0, dy)
        dst_y0, dst_y1 = max(0, dy), h - max(0, -dy)
        shifted[dst_y0:dst_y1, dst_x0:dst_x1] = alpha[src_y0:src_y1, src_x0:src_x1]
        outline_mask |= shifted & ~alpha

    if dithered:
        ys, xs = np.mgrid[0:h, 0:w]
        checker = (xs + ys) % 2 == 0
        outline_mask &= checker

    out = np.zeros_like(image)
    out[outline_mask] = color
    return out


def apply_color_ramp_shift(
    image: np.ndarray,
    ramp: list[tuple[int, int, int, int]],
    shift: int,
) -> np.ndarray:
    """Shift each pixel's color along a defined ramp (light->dark ordered palette list)
    by `shift` steps (positive = darker/later in ramp, negative = lighter/earlier).
    Pixels whose color is not found in the ramp are left unchanged.
    """
    if not ramp:
        return image.copy()
    ramp_arr = np.array(ramp, dtype=np.uint8)
    out = image.copy()
    h, w = image.shape[:2]
    flat = image.reshape(-1, 4)
    out_flat = out.reshape(-1, 4)

    for i, rgba in enumerate(ramp):
        matches = np.all(flat[:, :3] == np.array(rgba[:3], dtype=np.uint8), axis=-1) & (flat[:, 3] > 0)
        if not matches.any():
            continue
        new_idx = min(max(i + shift, 0), len(ramp) - 1)
        out_flat[matches, :3] = ramp_arr[new_idx, :3]
    return out


def auto_clean_double_pixels(image: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """Detect and clean stray diagonal 'double pixel' artifacts (a 2x2 block forming
    an L-shaped diagonal stair-step with a hole) across the image or within `mask`.
    Removes the redundant corner pixel that creates a jagged double-width diagonal.
    """
    h, w = image.shape[:2]
    out = image.copy()
    alpha = image[..., 3] > 0
    region = mask if mask is not None else np.ones((h, w), dtype=bool)

    for y in range(h - 1):
        for x in range(w - 1):
            if not (region[y, x] and region[y + 1, x + 1]):
                continue
            a = alpha[y, x]
            b = alpha[y, x + 1]
            c = alpha[y + 1, x]
            d = alpha[y + 1, x + 1]
            # Diagonal pattern: opposite corners filled, adjacent corners empty ->
            # this is a legitimate 1px diagonal, leave it. The "double" artifact is
            # when THREE of the four are filled (an L), which thickens the diagonal.
            filled_count = sum([a, b, c, d])
            if filled_count == 3:
                if a and b and c and not d:
                    out[y, x + 1] = 0
                elif a and b and d and not c:
                    out[y, x] = 0
                elif a and c and d and not b:
                    out[y + 1, x + 1] = 0
                elif b and c and d and not a:
                    out[y, x] = 0
    return out


def _box_sum(arr: np.ndarray, radius: int, axis: int) -> np.ndarray:
    """Sliding-window sum of width (2*radius+1) along `axis`, via padded cumsum."""
    pad_width = [(0, 0)] * arr.ndim
    pad_width[axis] = (radius, radius)
    padded = np.pad(arr, pad_width, mode="edge")
    csum = np.cumsum(padded, axis=axis)
    zero_shape = list(csum.shape)
    zero_shape[axis] = 1
    csum = np.concatenate([np.zeros(zero_shape, dtype=csum.dtype), csum], axis=axis)
    n = arr.shape[axis]
    hi = [slice(None)] * arr.ndim
    lo = [slice(None)] * arr.ndim
    hi[axis] = slice(2 * radius + 1, 2 * radius + 1 + n)
    lo[axis] = slice(0, n)
    return csum[tuple(hi)] - csum[tuple(lo)]


def box_blur(image: np.ndarray, radius: int = 1) -> np.ndarray:
    """Alpha-aware box blur: colors are premultiplied by alpha before averaging
    and un-premultiplied afterward, so transparent neighbors don't darken edges.
    """
    if radius <= 0:
        return image.copy()
    alpha = image[..., 3:4].astype(np.float32)
    premult_rgb = image[..., :3].astype(np.float32) * (alpha / 255.0)

    rgb_sum = _box_sum(_box_sum(premult_rgb, radius, axis=0), radius, axis=1)
    alpha_sum = _box_sum(_box_sum(alpha, radius, axis=0), radius, axis=1)
    kernel_area = float((2 * radius + 1) ** 2)

    out_alpha = alpha_sum / kernel_area
    safe_alpha_sum = np.where(alpha_sum == 0, 1.0, alpha_sum)
    out_rgb = rgb_sum / safe_alpha_sum * 255.0

    out = np.concatenate([out_rgb, out_alpha], axis=-1)
    return np.clip(out + 0.5, 0, 255).astype(np.uint8)


def sharpen(image: np.ndarray, amount: float = 1.0, radius: int = 1) -> np.ndarray:
    """Unsharp-mask sharpening: pushes each pixel away from its local blurred
    average, boosting edge contrast (useful for crisping up hand-painted shading).
    """
    blurred = box_blur(image, radius).astype(np.float32)
    orig = image.astype(np.float32)
    sharpened_rgb = orig[..., :3] + amount * (orig[..., :3] - blurred[..., :3])
    out = image.copy()
    out[..., :3] = np.clip(sharpened_rgb + 0.5, 0, 255).astype(np.uint8)
    out[image[..., 3] == 0] = 0
    return out


def adjust_brightness_contrast(image: np.ndarray, brightness: int = 0, contrast: int = 0) -> np.ndarray:
    """Shift brightness (-255..255) and contrast (-255..255) using the standard
    Photoshop-style contrast formula. Alpha is left untouched."""
    rgb = image[..., :3].astype(np.float32) + brightness
    contrast = max(-255, min(255, contrast))
    factor = (259.0 * (contrast + 255.0)) / (255.0 * (259.0 - contrast))
    rgb = factor * (rgb - 128.0) + 128.0
    out = image.copy()
    out[..., :3] = np.clip(rgb + 0.5, 0, 255).astype(np.uint8)
    out[image[..., 3] == 0] = 0
    return out


def invert_colors(image: np.ndarray) -> np.ndarray:
    """Invert RGB channels, leaving alpha untouched."""
    out = image.copy()
    out[..., :3] = 255 - out[..., :3]
    out[image[..., 3] == 0] = 0
    return out


def grayscale(image: np.ndarray) -> np.ndarray:
    """Desaturate to luminance (Rec. 601 weights), preserving alpha."""
    rgb = image[..., :3].astype(np.float32)
    luma = rgb[..., 0] * 0.299 + rgb[..., 1] * 0.587 + rgb[..., 2] * 0.114
    out = image.copy()
    out[..., :3] = np.clip(luma + 0.5, 0, 255).astype(np.uint8)[..., None]
    out[image[..., 3] == 0] = 0
    return out


def apply_drop_shadow(
    image: np.ndarray,
    offset: tuple[int, int] = (2, 2),
    color: tuple[int, int, int, int] = (0, 0, 0, 160),
) -> np.ndarray:
    """Return a new image with a drop shadow composited beneath `image`'s silhouette."""
    h, w = image.shape[:2]
    dx, dy = offset
    alpha = image[..., 3] > 0
    shadow_alpha = np.zeros((h, w), dtype=bool)

    src_x0, src_x1 = max(0, -dx), w - max(0, dx)
    dst_x0, dst_x1 = max(0, dx), w - max(0, -dx)
    src_y0, src_y1 = max(0, -dy), h - max(0, dy)
    dst_y0, dst_y1 = max(0, dy), h - max(0, -dy)
    if src_x0 < src_x1 and src_y0 < src_y1:
        shadow_alpha[dst_y0:dst_y1, dst_x0:dst_x1] = alpha[src_y0:src_y1, src_x0:src_x1]

    shadow_layer = np.zeros_like(image)
    shadow_layer[shadow_alpha] = color

    # Composite image over shadow (straight alpha over).
    top_a = (image[..., 3:4].astype(np.float32)) / 255.0
    bot = shadow_layer.astype(np.float32)
    top = image.astype(np.float32)
    out_a = top_a + bot[..., 3:4] / 255.0 * (1 - top_a)
    safe_a = np.where(out_a == 0, 1.0, out_a)
    out_rgb = (top[..., :3] * top_a + bot[..., :3] * (bot[..., 3:4] / 255.0) * (1 - top_a)) / safe_a
    result = np.concatenate([out_rgb, out_a * 255.0], axis=-1)
    return np.clip(result + 0.5, 0, 255).astype(np.uint8)
