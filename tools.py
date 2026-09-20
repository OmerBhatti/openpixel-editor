"""Toolset: pixel-perfect pencil, eraser, bucket fill, selections, eyedropper.

Tools operate on a Layer's pixel buffer (np.ndarray HxWx4 uint8) and are driven by
the canvas widget via mouse events translated to pixel coordinates.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto

import numpy as np

from brushes import Brush, stamp, erase as brush_erase


def bresenham_line(x0: int, y0: int, x1: int, y1: int) -> list[tuple[int, int]]:
    """Classic Bresenham line rasterization, returns list of (x, y) pixel coords."""
    points = []
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    x, y = x0, y0
    while True:
        points.append((x, y))
        if x == x1 and y == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x += sx
        if e2 <= dx:
            err += dx
            y += sy
    return points


def remove_diagonal_doubles(points: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Pixel-perfect pencil: when a line stair-steps diagonally, drop the redundant
    'elbow' pixel that creates a 2px-thick diagonal, keeping single-pixel-wide diagonals.
    """
    if len(points) < 3:
        return points
    cleaned = [points[0]]
    i = 1
    while i < len(points) - 1:
        prev = cleaned[-1]
        cur = points[i]
        nxt = points[i + 1]
        # Detect an "L" shaped step (horizontal then vertical, or vice versa) that forms
        # a 2x2 block with the next point -> drop current, keeping the diagonal single-wide.
        if (cur[0] - prev[0] != 0 and cur[1] - prev[1] == 0 and
                nxt[0] - cur[0] == 0 and nxt[1] - cur[1] != 0):
            i += 1
            continue
        if (cur[0] - prev[0] == 0 and cur[1] - prev[1] != 0 and
                nxt[0] - cur[0] != 0 and nxt[1] - cur[1] == 0):
            i += 1
            continue
        cleaned.append(cur)
        i += 1
    cleaned.append(points[-1])
    return cleaned


def set_pixel(pixels: np.ndarray, x: int, y: int, color: tuple[int, int, int, int]) -> bool:
    h, w = pixels.shape[:2]
    if 0 <= x < w and 0 <= y < h:
        pixels[y, x] = color
        return True
    return False


def flood_fill(
    pixels: np.ndarray,
    x: int,
    y: int,
    fill_color: tuple[int, int, int, int],
    tolerance: int = 0,
    layers_sample: np.ndarray | None = None,
) -> np.ndarray:
    """Flood fill starting at (x, y). Returns a boolean mask of filled pixels.

    `layers_sample` (optional) is a composite buffer used to determine matching regions
    when "sample all layers" is enabled, while the fill itself is still written to `pixels`.
    """
    h, w = pixels.shape[:2]
    if not (0 <= x < w and 0 <= y < h):
        return np.zeros((h, w), dtype=bool)

    sample_source = layers_sample if layers_sample is not None else pixels
    target_color = sample_source[y, x].astype(np.int32)

    diff = np.abs(sample_source.astype(np.int32) - target_color[None, None, :])
    within_tol = np.all(diff <= tolerance, axis=-1)

    # Flood fill via scanline using a stack, constrained to within_tol connectivity.
    visited = np.zeros((h, w), dtype=bool)
    stack = [(x, y)]
    while stack:
        cx, cy = stack.pop()
        if visited[cy, cx] or not within_tol[cy, cx]:
            continue
        # Extend left/right on this row.
        lx = cx
        while lx > 0 and within_tol[cy, lx - 1] and not visited[cy, lx - 1]:
            lx -= 1
        rx = cx
        while rx < w - 1 and within_tol[cy, rx + 1] and not visited[cy, rx + 1]:
            rx += 1
        visited[cy, lx:rx + 1] = True
        for nx in range(lx, rx + 1):
            if cy > 0 and within_tol[cy - 1, nx] and not visited[cy - 1, nx]:
                stack.append((nx, cy - 1))
            if cy < h - 1 and within_tol[cy + 1, nx] and not visited[cy + 1, nx]:
                stack.append((nx, cy + 1))

    pixels[visited] = fill_color
    return visited


def eyedropper(composite: np.ndarray, x: int, y: int) -> tuple[int, int, int, int] | None:
    h, w = composite.shape[:2]
    if 0 <= x < w and 0 <= y < h:
        return tuple(int(c) for c in composite[y, x])
    return None


# ---- Selections ----

@dataclass
class Selection:
    """A pixel-precise selection mask (H, W bool)."""
    mask: np.ndarray

    @classmethod
    def empty(cls, width: int, height: int) -> "Selection":
        return cls(mask=np.zeros((height, width), dtype=bool))

    @classmethod
    def rectangle(cls, width: int, height: int, x0: int, y0: int, x1: int, y1: int) -> "Selection":
        mask = np.zeros((height, width), dtype=bool)
        xs, xe = sorted((max(0, min(x0, width)), max(0, min(x1, width))))
        ys, ye = sorted((max(0, min(y0, height)), max(0, min(y1, height))))
        mask[ys:ye, xs:xe] = True
        return cls(mask=mask)

    @classmethod
    def lasso(cls, width: int, height: int, points: list[tuple[int, int]]) -> "Selection":
        """Freehand lasso selection via scanline polygon fill (even-odd rule)."""
        mask = np.zeros((height, width), dtype=bool)
        if len(points) < 3:
            return cls(mask=mask)
        ys = [p[1] for p in points]
        y_min, y_max = max(0, min(ys)), min(height - 1, max(ys))
        n = len(points)
        for y in range(y_min, y_max + 1):
            xs_hits = []
            for i in range(n):
                x0, y0 = points[i]
                x1, y1 = points[(i + 1) % n]
                if y0 == y1:
                    continue
                if (y0 <= y < y1) or (y1 <= y < y0):
                    t = (y - y0) / (y1 - y0)
                    xs_hits.append(x0 + t * (x1 - x0))
            xs_hits.sort()
            for i in range(0, len(xs_hits) - 1, 2):
                x_start = max(0, int(round(xs_hits[i])))
                x_end = min(width, int(round(xs_hits[i + 1])))
                if x_start < x_end:
                    mask[y, x_start:x_end] = True
        return cls(mask=mask)

    def bounds(self) -> tuple[int, int, int, int] | None:
        ys, xs = np.where(self.mask)
        if len(xs) == 0:
            return None
        return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1

    def is_empty(self) -> bool:
        return not self.mask.any()


class ToolKind(Enum):
    PENCIL = auto()
    BRUSH = auto()
    ERASER = auto()
    BUCKET = auto()
    SELECT_RECT = auto()
    SELECT_LASSO = auto()
    EYEDROPPER = auto()
    HAND = auto()
    ZOOM = auto()


class Tool(ABC):
    kind: ToolKind

    @abstractmethod
    def begin(self, ctx: "ToolContext", x: int, y: int) -> None: ...

    @abstractmethod
    def move(self, ctx: "ToolContext", x: int, y: int) -> None: ...

    @abstractmethod
    def end(self, ctx: "ToolContext", x: int, y: int) -> None: ...


@dataclass
class ToolContext:
    """Shared state passed to tools during a stroke."""
    pixels: np.ndarray
    color: tuple[int, int, int, int]
    brush: Brush
    pixel_perfect: bool = True
    tolerance: int = 0
    sample_all: bool = False
    composite: np.ndarray | None = None
    selection: Selection | None = None
    dirty_rects: list[tuple[int, int, int, int]] = field(default_factory=list)
    stroke_points: list[tuple[int, int]] = field(default_factory=list)
    last_point: tuple[int, int] | None = None
    picked_color: tuple[int, int, int, int] | None = None


class PencilTool(Tool):
    kind = ToolKind.PENCIL

    def __init__(self, erase: bool = False):
        self.erase = erase

    def _paint(self, ctx: ToolContext, x: int, y: int) -> tuple[int, int, int, int] | None:
        if self.erase:
            return brush_erase(ctx.pixels, ctx.brush, x, y)
        return stamp(ctx.pixels, ctx.brush, x, y, ctx.color)

    def _apply_brush_line(self, ctx: ToolContext, p0: tuple[int, int], p1: tuple[int, int]) -> None:
        raw_points = bresenham_line(p0[0], p0[1], p1[0], p1[1])
        points = remove_diagonal_doubles(raw_points) if ctx.pixel_perfect else raw_points
        for (x, y) in points:
            rect = self._paint(ctx, x, y)
            if rect:
                ctx.dirty_rects.append(rect)

    def begin(self, ctx: ToolContext, x: int, y: int) -> None:
        ctx.stroke_points = [(x, y)]
        ctx.last_point = (x, y)
        rect = self._paint(ctx, x, y)
        if rect:
            ctx.dirty_rects.append(rect)

    def move(self, ctx: ToolContext, x: int, y: int) -> None:
        if ctx.last_point is None:
            self.begin(ctx, x, y)
            return
        if (x, y) == ctx.last_point:
            return
        self._apply_brush_line(ctx, ctx.last_point, (x, y))
        ctx.last_point = (x, y)
        ctx.stroke_points.append((x, y))

    def end(self, ctx: ToolContext, x: int, y: int) -> None:
        self.move(ctx, x, y)
        ctx.last_point = None


class EraserTool(PencilTool):
    kind = ToolKind.ERASER

    def __init__(self):
        super().__init__(erase=True)


class BrushTool(Tool):
    """Custom stamp/dither brush painting tool (continuous stamping along strokes)."""
    kind = ToolKind.BRUSH

    def begin(self, ctx: ToolContext, x: int, y: int) -> None:
        ctx.last_point = (x, y)
        rect = stamp(ctx.pixels, ctx.brush, x, y, ctx.color)
        if rect:
            ctx.dirty_rects.append(rect)

    def move(self, ctx: ToolContext, x: int, y: int) -> None:
        if ctx.last_point is None:
            self.begin(ctx, x, y)
            return
        for (px, py) in bresenham_line(ctx.last_point[0], ctx.last_point[1], x, y):
            rect = stamp(ctx.pixels, ctx.brush, px, py, ctx.color)
            if rect:
                ctx.dirty_rects.append(rect)
        ctx.last_point = (x, y)

    def end(self, ctx: ToolContext, x: int, y: int) -> None:
        self.move(ctx, x, y)
        ctx.last_point = None


class BucketTool(Tool):
    kind = ToolKind.BUCKET

    def begin(self, ctx: ToolContext, x: int, y: int) -> None:
        sample = ctx.composite if ctx.sample_all and ctx.composite is not None else None
        mask = flood_fill(ctx.pixels, x, y, ctx.color, ctx.tolerance, sample)
        if mask.any():
            ys, xs = np.where(mask)
            ctx.dirty_rects.append((int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1))

    def move(self, ctx: ToolContext, x: int, y: int) -> None:
        pass

    def end(self, ctx: ToolContext, x: int, y: int) -> None:
        pass


class EyedropperTool(Tool):
    kind = ToolKind.EYEDROPPER

    def begin(self, ctx: ToolContext, x: int, y: int) -> None:
        source = ctx.composite if ctx.composite is not None else ctx.pixels
        ctx.picked_color = eyedropper(source, x, y)

    def move(self, ctx: ToolContext, x: int, y: int) -> None:
        self.begin(ctx, x, y)

    def end(self, ctx: ToolContext, x: int, y: int) -> None:
        self.begin(ctx, x, y)


class SelectRectTool(Tool):
    kind = ToolKind.SELECT_RECT

    def begin(self, ctx: ToolContext, x: int, y: int) -> None:
        ctx.stroke_points = [(x, y)]

    def move(self, ctx: ToolContext, x: int, y: int) -> None:
        if not ctx.stroke_points:
            self.begin(ctx, x, y)
            return
        ctx.stroke_points = [ctx.stroke_points[0], (x, y)]
        (x0, y0), (x1, y1) = ctx.stroke_points
        h, w = ctx.pixels.shape[:2]
        ctx.selection = Selection.rectangle(w, h, x0, y0, x1 + 1, y1 + 1)

    def end(self, ctx: ToolContext, x: int, y: int) -> None:
        self.move(ctx, x, y)


class SelectLassoTool(Tool):
    kind = ToolKind.SELECT_LASSO

    def begin(self, ctx: ToolContext, x: int, y: int) -> None:
        ctx.stroke_points = [(x, y)]

    def move(self, ctx: ToolContext, x: int, y: int) -> None:
        if not ctx.stroke_points or ctx.stroke_points[-1] != (x, y):
            ctx.stroke_points.append((x, y))
        if len(ctx.stroke_points) >= 3:
            h, w = ctx.pixels.shape[:2]
            ctx.selection = Selection.lasso(w, h, ctx.stroke_points)

    def end(self, ctx: ToolContext, x: int, y: int) -> None:
        self.move(ctx, x, y)


class NullTool(Tool):
    """Placeholder for tools (Hand, Zoom) whose behavior is handled directly by the
    canvas's mouse events rather than through the paint-stroke ToolContext pipeline.
    """
    kind = ToolKind.HAND

    def begin(self, ctx: ToolContext, x: int, y: int) -> None:
        pass

    def move(self, ctx: ToolContext, x: int, y: int) -> None:
        pass

    def end(self, ctx: ToolContext, x: int, y: int) -> None:
        pass


TOOL_REGISTRY: dict[ToolKind, type[Tool]] = {
    ToolKind.PENCIL: PencilTool,
    ToolKind.BRUSH: BrushTool,
    ToolKind.ERASER: EraserTool,
    ToolKind.BUCKET: BucketTool,
    ToolKind.SELECT_RECT: SelectRectTool,
    ToolKind.SELECT_LASSO: SelectLassoTool,
    ToolKind.EYEDROPPER: EyedropperTool,
    ToolKind.HAND: NullTool,
    ToolKind.ZOOM: NullTool,
}


def create_tool(kind: ToolKind) -> Tool:
    return TOOL_REGISTRY[kind]()
