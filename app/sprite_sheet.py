"""Sprite sheet layout metadata: a shared canvas subdivided into equal-size frame
cells (single row, single column, or a full grid) with optional padding between
cells. Painting still happens on one ordinary layer stack; this module only
describes how that canvas is sliced into frames for the thicker frame-grid
overlay and per-frame navigation.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


class SheetLayout(Enum):
    SINGLE = "Single"
    LINEAR = "Linear (Horizontal)"
    VERTICAL = "Vertical"
    GRID = "Grid"


@dataclass
class SpriteSheetInfo:
    frame_width: int
    frame_height: int
    columns: int
    rows: int
    padding: int = 0
    layout: SheetLayout = SheetLayout.SINGLE
    # Optional exact per-column/per-row (start, end) pixel bounds, set when this
    # info came from detect_sprite_sheet_grid(). Real-world sheets are often
    # *almost* but not perfectly uniform (rows differing by a pixel here and
    # there); when present, these override the uniform frame_width/height/
    # padding arithmetic so the grid overlay and frame navigation line up with
    # what's actually in the image instead of drifting on later rows/columns.
    col_bounds: list[tuple[int, int]] | None = None
    row_bounds: list[tuple[int, int]] | None = None

    @property
    def frame_count(self) -> int:
        return self.columns * self.rows

    @property
    def canvas_width(self) -> int:
        return self.columns * self.frame_width + (self.columns - 1) * self.padding

    @property
    def canvas_height(self) -> int:
        return self.rows * self.frame_height + (self.rows - 1) * self.padding

    def frame_rect(self, col: int, row: int) -> tuple[int, int, int, int]:
        """Return (x0, y0, x1, y1) pixel bounds (x1/y1 exclusive) for frame (col, row)."""
        if self.col_bounds is not None and self.row_bounds is not None:
            x0, x1 = self.col_bounds[col]
            y0, y1 = self.row_bounds[row]
            return x0, y0, x1, y1
        x0 = col * (self.frame_width + self.padding)
        y0 = row * (self.frame_height + self.padding)
        return x0, y0, x0 + self.frame_width, y0 + self.frame_height

    def frame_at_pixel(self, x: int, y: int) -> tuple[int, int] | None:
        """Return the (col, row) of the frame containing pixel (x, y), or None if
        the point falls in the padding gutter or outside the sheet."""
        if x < 0 or y < 0:
            return None
        if self.col_bounds is not None and self.row_bounds is not None:
            col = next((i for i, (s, e) in enumerate(self.col_bounds) if s <= x < e), None)
            row = next((i for i, (s, e) in enumerate(self.row_bounds) if s <= y < e), None)
            if col is None or row is None:
                return None
            return col, row
        stride_x = self.frame_width + self.padding
        stride_y = self.frame_height + self.padding
        col, rem_x = divmod(x, stride_x)
        row, rem_y = divmod(y, stride_y)
        if col >= self.columns or row >= self.rows:
            return None
        if rem_x >= self.frame_width or rem_y >= self.frame_height:
            return None  # inside the padding gutter
        return int(col), int(row)

    def to_dict(self) -> dict:
        return {
            "frame_width": self.frame_width,
            "frame_height": self.frame_height,
            "columns": self.columns,
            "rows": self.rows,
            "padding": self.padding,
            "layout": self.layout.value,
            "col_bounds": self.col_bounds,
            "row_bounds": self.row_bounds,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SpriteSheetInfo":
        layout = next((l for l in SheetLayout if l.value == data.get("layout")), SheetLayout.SINGLE)
        col_bounds = data.get("col_bounds")
        row_bounds = data.get("row_bounds")
        return cls(
            frame_width=data["frame_width"],
            frame_height=data["frame_height"],
            columns=data["columns"],
            rows=data["rows"],
            padding=data.get("padding", 0),
            layout=layout,
            col_bounds=[tuple(b) for b in col_bounds] if col_bounds else None,
            row_bounds=[tuple(b) for b in row_bounds] if row_bounds else None,
        )


def _content_runs_and_gutters(empty_mask: np.ndarray) -> tuple[list[tuple[int, int]], list[int]]:
    """Scan a 1D boolean "is empty/gutter" mask and return the contiguous
    non-empty (content) runs as (start, end) pairs, plus the lengths of the
    gutter runs found strictly *between* two content runs (leading/trailing
    empty margins are not counted as inter-frame padding).
    """
    n = len(empty_mask)
    content_runs: list[tuple[int, int]] = []
    gutter_lengths: list[int] = []
    i = 0
    while i < n:
        if not empty_mask[i]:
            j = i
            while j < n and not empty_mask[j]:
                j += 1
            content_runs.append((i, j))
            i = j
        else:
            j = i
            while j < n and empty_mask[j]:
                j += 1
            if content_runs:
                gutter_lengths.append(j - i)
            i = j
    return content_runs, gutter_lengths


def _most_common(values: list[int]) -> int | None:
    if not values:
        return None
    return max(set(values), key=values.count)


def _suppress_spurious_gutters(empty_mask: np.ndarray, min_ratio: float = 0.6) -> np.ndarray:
    """Real sprites occasionally have an internal near-background band (a light
    hairline, a collar, shading) that a naive scan mistakes for an inter-frame
    gutter, splitting one frame's row/column into two. Such artifacts can even
    outnumber the genuine gutters (an N-row sheet has only N-1 real row gutters
    but can have N spurious ones, one per frame), so picking the *most common*
    gutter length as "typical" is unreliable — it can lock onto the noise.
    Deliberate inter-frame padding is instead almost always the *widest*
    recurring gap, so any gutter noticeably shorter than the widest one found
    is treated as a false positive and flipped back to "content".
    """
    runs, gutters = _content_runs_and_gutters(empty_mask)
    if len(runs) < 2 or not gutters:
        return empty_mask
    typical = max(gutters)
    threshold = typical * min_ratio

    refined = empty_mask.copy()
    n = len(empty_mask)
    i = 0
    seen_content = False
    while i < n:
        if not empty_mask[i]:
            j = i
            while j < n and not empty_mask[j]:
                j += 1
            seen_content = True
            i = j
        else:
            j = i
            while j < n and empty_mask[j]:
                j += 1
            is_trailing = j == n
            if seen_content and not is_trailing and (j - i) < threshold:
                refined[i:j] = False
            i = j
    return refined


def detect_sprite_sheet_grid(image: np.ndarray, tolerance: float = 0.03) -> SpriteSheetInfo | None:
    """Best-effort auto-detection of a sprite sheet's frame size, padding, and
    grid dimensions from gutter columns/rows: pixel columns/rows that are
    almost entirely transparent (or, for a sheet with no transparency, almost
    entirely the sheet's background color) are treated as gaps between frames.

    A small `tolerance` (fraction of non-background pixels still allowed in a
    "gutter" column/row) makes this tolerant of sprites whose hair, limbs, etc.
    stray slightly past their nominal frame box and touch a neighboring column
    or row — a strict all-or-nothing emptiness test breaks on real art like that.

    Returns None when no clear, uniform grid can be inferred (e.g. frames butt
    up against each other with zero padding and nothing separates them), so the
    caller can fall back to asking the user or a 1x1 default.
    """
    alpha = image[..., 3]
    if np.any(alpha == 0):
        empty_px = alpha == 0
    else:
        # No transparency at all: assume the sheet's most common pixel color is
        # its background/gutter fill (more robust than the corner pixel, which
        # can itself be sprite content in a densely-packed sheet).
        flat = image.reshape(-1, 4)
        colors, counts = np.unique(flat, axis=0, return_counts=True)
        bg = colors[np.argmax(counts)]
        empty_px = np.all(image == bg, axis=-1)

    # Fraction of non-background pixels per column/row; treat as a gutter when
    # that fraction is small enough to be stray overlap rather than real content.
    col_occupancy = 1.0 - np.mean(empty_px, axis=0)
    row_occupancy = 1.0 - np.mean(empty_px, axis=1)
    col_empty = col_occupancy <= tolerance
    row_empty = row_occupancy <= tolerance

    # Drop short spurious "gutters" (a stray light-colored band inside a sprite
    # that isn't real inter-frame padding) before extracting the final runs.
    col_empty = _suppress_spurious_gutters(col_empty)
    row_empty = _suppress_spurious_gutters(row_empty)

    col_runs, col_gutters = _content_runs_and_gutters(col_empty)
    row_runs, row_gutters = _content_runs_and_gutters(row_empty)
    if not col_runs or not row_runs:
        return None

    cols, rows = len(col_runs), len(row_runs)
    padding = _most_common(col_gutters + row_gutters) or 0

    # Representative frame size for the editable Width/Height fields (real
    # sheets are often *almost* uniform); the exact per-frame col/row bounds
    # below are what actually drive the grid overlay and frame navigation, so
    # small per-row/column variance doesn't cause drift on later frames.
    frame_w = _most_common([end - start for start, end in col_runs]) or (col_runs[0][1] - col_runs[0][0])
    frame_h = _most_common([end - start for start, end in row_runs]) or (row_runs[0][1] - row_runs[0][0])
    if frame_w <= 0 or frame_h <= 0:
        return None

    layout = SheetLayout.GRID if (cols > 1 and rows > 1) else (
        SheetLayout.LINEAR if rows == 1 and cols > 1 else
        SheetLayout.VERTICAL if cols == 1 and rows > 1 else
        SheetLayout.SINGLE
    )
    return SpriteSheetInfo(
        frame_width=frame_w, frame_height=frame_h,
        columns=cols, rows=rows, padding=padding, layout=layout,
        col_bounds=col_runs, row_bounds=row_runs,
    )
