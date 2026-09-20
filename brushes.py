"""Custom stamp & dither brush engine. Brushes are square/rect NumPy uint8 matrices
(alpha masks, 0-255) that are stamped onto the canvas at a given color. Persisted as
.pxbrush JSON files.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Brush:
    name: str
    matrix: np.ndarray  # (H, W) uint8, 0-255 coverage/alpha mask
    kind: str = "stamp"  # "stamp" or "dither"

    @property
    def height(self) -> int:
        return self.matrix.shape[0]

    @property
    def width(self) -> int:
        return self.matrix.shape[1]

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "kind": self.kind,
            "width": self.width,
            "height": self.height,
            "matrix": self.matrix.tolist(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Brush":
        matrix = np.array(data["matrix"], dtype=np.uint8)
        return cls(name=data.get("name", "Brush"), matrix=matrix, kind=data.get("kind", "stamp"))

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "Brush":
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    @classmethod
    def solid_square(cls, size: int = 1, name: str | None = None) -> "Brush":
        return cls(name=name or f"Square {size}px", matrix=np.full((size, size), 255, dtype=np.uint8))

    @classmethod
    def circle(cls, diameter: int, name: str | None = None) -> "Brush":
        r = diameter / 2.0
        ys, xs = np.mgrid[0:diameter, 0:diameter]
        cx = cy = (diameter - 1) / 2.0
        dist = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
        mask = (dist <= r).astype(np.uint8) * 255
        return cls(name=name or f"Circle {diameter}px", matrix=mask)


# ---- Built-in ordered dither pattern generators (Bayer matrices) ----

_BAYER_2 = np.array([[0, 2], [3, 1]], dtype=np.float32)


def _bayer_matrix(n: int) -> np.ndarray:
    """Recursively build an n x n Bayer matrix (n must be a power of two)."""
    if n == 2:
        return _BAYER_2.copy()
    smaller = _bayer_matrix(n // 2)
    size = n // 2
    top_left = 4 * smaller
    top_right = 4 * smaller + 2
    bottom_left = 4 * smaller + 3
    bottom_right = 4 * smaller + 1
    top = np.hstack([top_left, top_right])
    bottom = np.hstack([bottom_left, bottom_right])
    return np.vstack([top, bottom])


def make_dither_brush(size: int, coverage: float = 0.5, name: str | None = None) -> Brush:
    """Build a dither brush (2x2, 4x4, or 8x8) thresholded at `coverage` (0..1)."""
    if size not in (2, 4, 8):
        raise ValueError("Dither brush size must be 2, 4, or 8")
    bayer = _bayer_matrix(size)
    normalized = bayer / (size * size)
    mask = (normalized < coverage).astype(np.uint8) * 255
    return Brush(name=name or f"Dither {size}x{size} ({int(coverage * 100)}%)", matrix=mask, kind="dither")


class BrushManager:
    """Holds the set of loaded/created brush presets."""

    def __init__(self):
        self.brushes: list[Brush] = [
            Brush.solid_square(1),
            Brush.solid_square(2),
            Brush.solid_square(4),
            Brush.circle(6),
            make_dither_brush(2, 0.5),
            make_dither_brush(4, 0.5),
            make_dither_brush(8, 0.5),
        ]
        self.active_index: int = 0

    @property
    def active(self) -> Brush:
        return self.brushes[self.active_index]

    def add(self, brush: Brush) -> int:
        self.brushes.append(brush)
        return len(self.brushes) - 1

    def remove(self, index: int) -> None:
        if len(self.brushes) > 1:
            del self.brushes[index]
            self.active_index = min(self.active_index, len(self.brushes) - 1)


def stamp(
    pixels: np.ndarray,
    brush: Brush,
    cx: int,
    cy: int,
    color: tuple[int, int, int, int],
) -> tuple[int, int, int, int] | None:
    """Stamp `brush` centered at (cx, cy) onto `pixels` (H,W,4 uint8) in place.

    Returns the affected bounding box (x0, y0, x1, y1) exclusive, or None if fully clipped.
    """
    h, w = pixels.shape[:2]
    bh, bw = brush.height, brush.width
    x0 = cx - bw // 2
    y0 = cy - bh // 2
    x1 = x0 + bw
    y1 = y0 + bh

    src_x0, src_y0 = 0, 0
    if x0 < 0:
        src_x0 = -x0
        x0 = 0
    if y0 < 0:
        src_y0 = -y0
        y0 = 0
    src_x1 = bw - max(0, x1 - w)
    src_y1 = bh - max(0, y1 - h)
    x1 = min(x1, w)
    y1 = min(y1, h)

    if x0 >= x1 or y0 >= y1 or src_x0 >= src_x1 or src_y0 >= src_y1:
        return None

    mask = brush.matrix[src_y0:src_y1, src_x0:src_x1].astype(np.float32) / 255.0
    region = pixels[y0:y1, x0:x1].astype(np.float32)
    color_arr = np.array(color, dtype=np.float32)

    alpha = mask[..., None] * (color_arr[3] / 255.0)
    out_a = alpha + region[..., 3:4] / 255.0 * (1 - alpha)
    safe_a = np.where(out_a == 0, 1.0, out_a)
    out_rgb = (color_arr[:3] * alpha + region[..., :3] * (region[..., 3:4] / 255.0) * (1 - alpha)) / safe_a

    region[..., :3] = out_rgb
    region[..., 3:4] = out_a * 255.0
    pixels[y0:y1, x0:x1] = np.clip(region + 0.5, 0, 255).astype(np.uint8)
    return (x0, y0, x1, y1)


def erase(
    pixels: np.ndarray,
    brush: Brush,
    cx: int,
    cy: int,
) -> tuple[int, int, int, int] | None:
    """Erase using `brush` as a coverage mask centered at (cx, cy): directly reduces
    alpha (rather than alpha-compositing a transparent color, which is a no-op).

    Returns the affected bounding box (x0, y0, x1, y1) exclusive, or None if fully clipped.
    """
    h, w = pixels.shape[:2]
    bh, bw = brush.height, brush.width
    x0 = cx - bw // 2
    y0 = cy - bh // 2
    x1 = x0 + bw
    y1 = y0 + bh

    src_x0, src_y0 = 0, 0
    if x0 < 0:
        src_x0 = -x0
        x0 = 0
    if y0 < 0:
        src_y0 = -y0
        y0 = 0
    src_x1 = bw - max(0, x1 - w)
    src_y1 = bh - max(0, y1 - h)
    x1 = min(x1, w)
    y1 = min(y1, h)

    if x0 >= x1 or y0 >= y1 or src_x0 >= src_x1 or src_y0 >= src_y1:
        return None

    mask = brush.matrix[src_y0:src_y1, src_x0:src_x1].astype(np.float32) / 255.0
    region = pixels[y0:y1, x0:x1]
    new_alpha = region[..., 3].astype(np.float32) * (1.0 - mask)
    region[..., 3] = np.clip(new_alpha + 0.5, 0, 255).astype(np.uint8)
    fully_cleared = region[..., 3] == 0
    region[fully_cleared, :3] = 0
    return (x0, y0, x1, y1)
