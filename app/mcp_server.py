"""OpenPixel MCP server: exposes pixel-art creation/editing as MCP tools so
other agents can drive OpenPixel programmatically, headlessly (no GUI window).

Sessions hold an in-memory canvas (LayerStack + Palette + optional sprite sheet
layout), addressed by a session_id returned from whichever tool created them
(create_canvas / import_image_file / open_project_file / import_sprite_sheet_file).
Run directly for local (stdio) use with an MCP-capable agent/client:

    python mcp_server.py
"""
from __future__ import annotations

import base64
import os
import sys
import uuid
from dataclasses import dataclass, field

import numpy as np
from PySide6.QtGui import QGuiApplication

from mcp.server.fastmcp import FastMCP, Image

from layers import LayerStack, BlendMode, Layer
from palette import Palette
from brushes import Brush, BrushManager, make_dither_brush, stamp as brush_stamp
from tools import bresenham_line, remove_diagonal_doubles, flood_fill as _flood_fill
import filters as flt
from transform import transform_selection
from sprite_sheet import SpriteSheetInfo, SheetLayout, detect_sprite_sheet_grid
from project_io import save_project as _save_project_file, load_project as _load_project_file
from utils.image_io import (
    png_bytes as _png_bytes,
    save_rgba_png as _save_png_file,
    load_rgba_image as _load_image_file_or_none,
    read_sprite_sheet_metadata as _read_sprite_sheet_metadata,
)

# QImage's PNG/JPEG/WebP codecs need a live QGuiApplication for plugin loading,
# even though this server never creates a window.
_qt_app = QGuiApplication.instance() or QGuiApplication(sys.argv[:1])

mcp = FastMCP("OpenPixel", instructions=(
    "Create and edit pixel art. Start with create_canvas, import_image_file, "
    "open_project_file, or import_sprite_sheet_file to get a session_id, then "
    "use the drawing/layer/filter/palette tools with that id. Call get_image "
    "after edits to see the current result before deciding the next step."
))

BACKGROUND_COLORS = {
    "transparent": (0, 0, 0, 0),
    "white": (255, 255, 255, 255),
    "black": (0, 0, 0, 255),
}


@dataclass
class Session:
    layer_stack: LayerStack
    palette: Palette = field(default_factory=Palette.default)
    sprite_sheet: SpriteSheetInfo | None = None
    name: str = "Untitled"
    clipboard: np.ndarray | None = None


SESSIONS: dict[str, Session] = {}


def _new_session_id() -> str:
    return uuid.uuid4().hex[:12]


def _get_session(session_id: str) -> Session:
    session = SESSIONS.get(session_id)
    if session is None:
        raise ValueError(f"No such session_id: {session_id!r}. Create one first.")
    return session


def _get_layer(session: Session, layer_index: int | None) -> Layer:
    stack = session.layer_stack
    idx = stack.active_index if layer_index is None else layer_index
    if not (0 <= idx < len(stack.layers)):
        raise ValueError(f"layer_index {idx} out of range (0..{len(stack.layers) - 1})")
    return stack.layers[idx]


def _color(rgba: list[int]) -> tuple[int, int, int, int]:
    if len(rgba) == 3:
        r, g, b = rgba
        a = 255
    elif len(rgba) == 4:
        r, g, b, a = rgba
    else:
        raise ValueError("color must be [r,g,b] or [r,g,b,a], each 0-255")
    for c in (r, g, b, a):
        if not (0 <= c <= 255):
            raise ValueError("color channels must be in 0..255")
    return int(r), int(g), int(b), int(a)


def _clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


def _load_image_file(path: str) -> np.ndarray:
    arr = _load_image_file_or_none(path)
    if arr is None:
        raise ValueError(f"Couldn't read image file: {path}")
    return arr


def _session_summary(session_id: str, session: Session) -> dict:
    stack = session.layer_stack
    return {
        "session_id": session_id,
        "name": session.name,
        "width": stack.width,
        "height": stack.height,
        "layer_count": len(stack.layers),
        "active_layer": stack.active_index,
        "sprite_sheet": session.sprite_sheet.to_dict() if session.sprite_sheet else None,
    }


# ---------------------------------------------------------------------------
# Session lifecycle
# ---------------------------------------------------------------------------

@mcp.tool()
def create_canvas(width: int, height: int, background: str = "transparent", name: str = "Untitled") -> dict:
    """Create a new blank pixel-art canvas and return its session_id.

    background: "transparent", "white", or "black".
    """
    if background not in BACKGROUND_COLORS:
        raise ValueError("background must be 'transparent', 'white', or 'black'")
    if width < 1 or height < 1 or width > 4096 or height > 4096:
        raise ValueError("width/height must be between 1 and 4096")
    stack = LayerStack(width, height)
    stack.layers[0].pixels[:, :] = BACKGROUND_COLORS[background]
    session_id = _new_session_id()
    SESSIONS[session_id] = Session(layer_stack=stack, name=name)
    return _session_summary(session_id, SESSIONS[session_id])


@mcp.tool()
def import_image_file(path: str, name: str | None = None) -> dict:
    """Import a PNG/JPEG/WebP file from disk as a new session."""
    arr = _load_image_file(path)
    h, w = arr.shape[:2]
    stack = LayerStack(w, h)
    stack.layers[0].pixels = arr
    session_id = _new_session_id()
    session_name = name or os.path.splitext(os.path.basename(path))[0]
    SESSIONS[session_id] = Session(layer_stack=stack, name=session_name)
    return _session_summary(session_id, SESSIONS[session_id])


@mcp.tool()
def import_sprite_sheet_file(
    path: str,
    frame_width: int | None = None,
    frame_height: int | None = None,
    padding: int | None = None,
    name: str | None = None,
) -> dict:
    """Import a sprite sheet PNG/JPEG/WebP as a new session with frame markers.

    Layout resolution order: (1) exact layout embedded by OpenPixel's own PNG
    export, if present and no explicit frame_width/height override is given;
    (2) explicit frame_width/frame_height/padding if provided; (3) best-effort
    auto-detection from transparent/background gutters between frames.
    """
    arr = _load_image_file(path)
    h, w = arr.shape[:2]
    stack = LayerStack(w, h)
    stack.layers[0].pixels = arr

    sheet: SpriteSheetInfo | None = None
    embedded = _read_sprite_sheet_metadata(path)
    if embedded is not None and frame_width is None and frame_height is None:
        sheet = embedded
    elif frame_width is not None and frame_height is not None:
        pad = padding or 0
        cols = max(1, (w + pad) // (frame_width + pad))
        rows = max(1, (h + pad) // (frame_height + pad))
        layout = SheetLayout.GRID if (cols > 1 and rows > 1) else (
            SheetLayout.LINEAR if rows == 1 and cols > 1 else
            SheetLayout.VERTICAL if cols == 1 and rows > 1 else SheetLayout.SINGLE
        )
        sheet = SpriteSheetInfo(frame_width, frame_height, cols, rows, pad, layout=layout)
    else:
        sheet = detect_sprite_sheet_grid(arr)
        if sheet is None:
            raise ValueError(
                "Couldn't auto-detect a frame grid from this image. Pass explicit "
                "frame_width/frame_height (and padding) instead."
            )

    session_id = _new_session_id()
    session_name = name or os.path.splitext(os.path.basename(path))[0]
    SESSIONS[session_id] = Session(
        layer_stack=stack, name=session_name,
        sprite_sheet=sheet if sheet.frame_count > 1 else None,
    )
    return _session_summary(session_id, SESSIONS[session_id])


@mcp.tool()
def open_project_file(path: str) -> dict:
    """Open an OpenPixel .opxproj project file as a session."""
    stack, palette, name, sheet = _load_project_file(path)
    session_id = _new_session_id()
    SESSIONS[session_id] = Session(layer_stack=stack, palette=palette, sprite_sheet=sheet, name=name)
    return _session_summary(session_id, SESSIONS[session_id])


@mcp.tool()
def save_project_file(session_id: str, path: str) -> dict:
    """Save a session as an OpenPixel .opxproj project file (preserves layers,
    palette, and sprite sheet layout)."""
    session = _get_session(session_id)
    if not path.lower().endswith(".opxproj"):
        path += ".opxproj"
    _save_project_file(path, session.layer_stack, session.palette, session.name, session.sprite_sheet)
    return {"path": path}


@mcp.tool()
def export_png_file(session_id: str, path: str) -> dict:
    """Export the session's flattened composite as a PNG file. If a sprite
    sheet layout is set, its exact frame geometry is embedded in the PNG so
    re-importing it later needs no re-detection."""
    session = _get_session(session_id)
    if not path.lower().endswith(".png"):
        path += ".png"
    _save_png_file(session.layer_stack.composite(), path, session.sprite_sheet)
    return {"path": path}


@mcp.tool()
def close_session(session_id: str) -> dict:
    """Free a session's memory once you're done with it."""
    existed = SESSIONS.pop(session_id, None) is not None
    return {"closed": existed}


@mcp.tool()
def list_sessions() -> list[dict]:
    """List every currently open session."""
    return [_session_summary(sid, s) for sid, s in SESSIONS.items()]


@mcp.tool()
def get_canvas_info(session_id: str) -> dict:
    """Get a session's canvas size, layer list, palette, and sprite sheet info."""
    session = _get_session(session_id)
    stack = session.layer_stack
    info = _session_summary(session_id, session)
    info["layers"] = [
        {
            "index": i, "name": layer.name, "opacity": layer.opacity,
            "visible": layer.visible, "blend_mode": layer.blend_mode.value, "locked": layer.locked,
        }
        for i, layer in enumerate(stack.layers)
    ]
    info["palette"] = list(session.palette.colors)
    return info


# ---------------------------------------------------------------------------
# Layers
# ---------------------------------------------------------------------------

@mcp.tool()
def add_layer(session_id: str, name: str | None = None) -> dict:
    """Add a new blank transparent layer above the active one and make it active."""
    session = _get_session(session_id)
    layer = session.layer_stack.add_layer(name)
    return {"index": session.layer_stack.active_index, "name": layer.name}


@mcp.tool()
def remove_layer(session_id: str, layer_index: int) -> dict:
    """Remove a layer by index (a canvas must always keep at least one layer)."""
    session = _get_session(session_id)
    session.layer_stack.remove_layer(layer_index)
    return {"layer_count": len(session.layer_stack.layers), "active_layer": session.layer_stack.active_index}


@mcp.tool()
def set_active_layer(session_id: str, layer_index: int) -> dict:
    """Set which layer subsequent drawing tools affect by default."""
    session = _get_session(session_id)
    if not (0 <= layer_index < len(session.layer_stack.layers)):
        raise ValueError("layer_index out of range")
    session.layer_stack.active_index = layer_index
    return {"active_layer": layer_index}


@mcp.tool()
def set_layer_properties(
    session_id: str,
    layer_index: int,
    opacity: float | None = None,
    visible: bool | None = None,
    blend_mode: str | None = None,
    name: str | None = None,
) -> dict:
    """Update a layer's opacity (0-1), visibility, blend mode
    ("Normal"/"Multiply"/"Screen"), and/or name. Only given fields change."""
    session = _get_session(session_id)
    layer = _get_layer(session, layer_index)
    if opacity is not None:
        layer.opacity = max(0.0, min(1.0, opacity))
    if visible is not None:
        layer.visible = visible
    if blend_mode is not None:
        try:
            layer.blend_mode = next(m for m in BlendMode if m.value == blend_mode)
        except StopIteration:
            raise ValueError(f"blend_mode must be one of {[m.value for m in BlendMode]}")
    if name is not None:
        layer.name = name
    return {
        "index": layer_index, "name": layer.name, "opacity": layer.opacity,
        "visible": layer.visible, "blend_mode": layer.blend_mode.value,
    }


@mcp.tool()
def clear_layer(session_id: str, layer_index: int | None = None) -> dict:
    """Erase a layer's contents to fully transparent (defaults to the active layer)."""
    session = _get_session(session_id)
    layer = _get_layer(session, layer_index)
    layer.pixels[:, :] = 0
    return {"cleared": True}


@mcp.tool()
def duplicate_layer(session_id: str, layer_index: int | None = None) -> dict:
    """Duplicate a layer (defaults to the active layer), inserting the copy
    above it and making the copy active."""
    session = _get_session(session_id)
    source = _get_layer(session, layer_index)
    new_layer = session.layer_stack.add_layer(f"{source.name} copy")
    new_layer.pixels = source.pixels.copy()
    new_layer.opacity = source.opacity
    new_layer.blend_mode = source.blend_mode
    return {"index": session.layer_stack.active_index, "name": new_layer.name}


@mcp.tool()
def move_layer(session_id: str, from_index: int, to_index: int) -> dict:
    """Reorder a layer in the stack (index 0 is bottom-most, last is top-most)."""
    session = _get_session(session_id)
    stack = session.layer_stack
    if not (0 <= from_index < len(stack.layers)) or not (0 <= to_index < len(stack.layers)):
        raise ValueError(f"indices must be within 0..{len(stack.layers) - 1}")
    stack.move_layer(from_index, to_index)
    return {"active_layer": stack.active_index}


@mcp.tool()
def resize_canvas(session_id: str, width: int, height: int) -> dict:
    """Resize the canvas, cropping or padding every layer from the top-left
    corner (existing content keeps its position; new area is transparent)."""
    session = _get_session(session_id)
    if width < 1 or height < 1 or width > 4096 or height > 4096:
        raise ValueError("width/height must be between 1 and 4096")
    session.layer_stack.resize_canvas(width, height)
    return {"width": width, "height": height}


# ---------------------------------------------------------------------------
# Drawing primitives (all act on layer_index, default: the active layer)
# ---------------------------------------------------------------------------

@mcp.tool()
def set_pixel(session_id: str, x: int, y: int, color: list[int], layer_index: int | None = None) -> dict:
    """Set a single pixel's color. color is [r,g,b] or [r,g,b,a] (0-255)."""
    session = _get_session(session_id)
    layer = _get_layer(session, layer_index)
    h, w = layer.pixels.shape[:2]
    if not (0 <= x < w and 0 <= y < h):
        raise ValueError(f"({x},{y}) is outside the {w}x{h} canvas")
    layer.pixels[y, x] = _color(color)
    return {"x": x, "y": y}


@mcp.tool()
def get_pixel(session_id: str, x: int, y: int, layer_index: int | None = None) -> dict:
    """Read a single pixel's RGBA color."""
    session = _get_session(session_id)
    layer = _get_layer(session, layer_index)
    h, w = layer.pixels.shape[:2]
    if not (0 <= x < w and 0 <= y < h):
        raise ValueError(f"({x},{y}) is outside the {w}x{h} canvas")
    return {"color": [int(c) for c in layer.pixels[y, x]]}


@mcp.tool()
def draw_line(
    session_id: str, x0: int, y0: int, x1: int, y1: int, color: list[int],
    thickness: int = 1, pixel_perfect: bool = True, layer_index: int | None = None,
) -> dict:
    """Draw a straight line with Bresenham rasterization. thickness > 1 stamps a
    square brush of that size along the line. pixel_perfect removes the
    redundant "double pixel" that a naive line leaves on 45-degree diagonals."""
    session = _get_session(session_id)
    layer = _get_layer(session, layer_index)
    rgba = _color(color)
    points = bresenham_line(x0, y0, x1, y1)
    if pixel_perfect and thickness == 1:
        points = remove_diagonal_doubles(points)
    brush = Brush.solid_square(max(1, thickness))
    for px, py in points:
        brush_stamp(layer.pixels, brush, px, py, rgba)
    return {"points_drawn": len(points)}


@mcp.tool()
def draw_rect(
    session_id: str, x0: int, y0: int, x1: int, y1: int, color: list[int],
    filled: bool = False, layer_index: int | None = None,
) -> dict:
    """Draw a rectangle from (x0,y0) to (x1,y1) inclusive, outlined or filled."""
    session = _get_session(session_id)
    layer = _get_layer(session, layer_index)
    h, w = layer.pixels.shape[:2]
    x0, x1 = sorted((x0, x1))
    y0, y1 = sorted((y0, y1))
    x0, y0 = _clamp(x0, 0, w - 1), _clamp(y0, 0, h - 1)
    x1, y1 = _clamp(x1, 0, w - 1), _clamp(y1, 0, h - 1)
    rgba = _color(color)
    if filled:
        layer.pixels[y0:y1 + 1, x0:x1 + 1] = rgba
    else:
        layer.pixels[y0, x0:x1 + 1] = rgba
        layer.pixels[y1, x0:x1 + 1] = rgba
        layer.pixels[y0:y1 + 1, x0] = rgba
        layer.pixels[y0:y1 + 1, x1] = rgba
    return {"x0": x0, "y0": y0, "x1": x1, "y1": y1}


@mcp.tool()
def draw_circle(
    session_id: str, cx: int, cy: int, radius: int, color: list[int],
    filled: bool = False, layer_index: int | None = None,
) -> dict:
    """Draw a circle centered at (cx, cy), outlined or filled."""
    session = _get_session(session_id)
    layer = _get_layer(session, layer_index)
    h, w = layer.pixels.shape[:2]
    rgba = _color(color)
    ys, xs = np.mgrid[0:h, 0:w]
    dist = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
    if filled:
        mask = dist <= radius + 0.5
    else:
        mask = np.abs(dist - radius) <= 0.5
    layer.pixels[mask] = rgba
    return {"pixels_set": int(mask.sum())}


@mcp.tool()
def stamp_brush(
    session_id: str, x: int, y: int, color: list[int], brush: str = "square:1",
    layer_index: int | None = None,
) -> dict:
    """Stamp a brush centered at (x, y). brush is one of: "square:N" (solid
    NxN, e.g. "square:3"), "circle:N" (solid circle of diameter N), or
    "dither:2"/"dither:4"/"dither:8" (a Bayer ordered-dither pattern at 50%
    coverage, useful for pixel-art shading/texture). For a fully custom shape,
    use draw_line/draw_rect/draw_circle/set_pixel instead.
    """
    session = _get_session(session_id)
    layer = _get_layer(session, layer_index)
    kind, _, size_str = brush.partition(":")
    size = int(size_str) if size_str else 1
    if kind == "square":
        brush_obj = Brush.solid_square(max(1, size))
    elif kind == "circle":
        brush_obj = Brush.circle(max(1, size))
    elif kind == "dither":
        if size not in (2, 4, 8):
            raise ValueError("dither brush size must be 2, 4, or 8")
        brush_obj = make_dither_brush(size, 0.5)
    else:
        raise ValueError('brush must look like "square:N", "circle:N", or "dither:2|4|8"')
    rect = brush_stamp(layer.pixels, brush_obj, x, y, _color(color))
    return {"affected_rect": list(rect) if rect else None}


@mcp.tool()
def flood_fill_area(
    session_id: str, x: int, y: int, color: list[int],
    tolerance: int = 0, sample_all_layers: bool = False, layer_index: int | None = None,
) -> dict:
    """Bucket-fill the contiguous region touching (x, y) that matches its
    starting color within `tolerance` (0-255 per channel)."""
    session = _get_session(session_id)
    layer = _get_layer(session, layer_index)
    sample = session.layer_stack.composite() if sample_all_layers else None
    mask = _flood_fill(layer.pixels, x, y, _color(color), tolerance, sample)
    return {"pixels_filled": int(mask.sum())}


# ---------------------------------------------------------------------------
# Regions (rectangle-based copy/paste/flip/rotate/transform; there's no
# freeform GUI-style selection object over MCP, only explicit coordinates)
# ---------------------------------------------------------------------------

def _clamp_rect(x0: int, y0: int, x1: int, y1: int, w: int, h: int) -> tuple[int, int, int, int]:
    x0, x1 = sorted((x0, x1))
    y0, y1 = sorted((y0, y1))
    x0, y0 = _clamp(x0, 0, w - 1), _clamp(y0, 0, h - 1)
    x1, y1 = _clamp(x1, 0, w - 1), _clamp(y1, 0, h - 1)
    return x0, y0, x1 + 1, y1 + 1  # exclusive end


@mcp.tool()
def copy_region(session_id: str, x0: int, y0: int, x1: int, y1: int, layer_index: int | None = None) -> dict:
    """Copy a rectangular region (x0,y0)-(x1,y1) inclusive from a layer to the
    session's clipboard, for use with paste_region."""
    session = _get_session(session_id)
    layer = _get_layer(session, layer_index)
    h, w = layer.pixels.shape[:2]
    cx0, cy0, cx1, cy1 = _clamp_rect(x0, y0, x1, y1, w, h)
    session.clipboard = layer.pixels[cy0:cy1, cx0:cx1].copy()
    return {"width": cx1 - cx0, "height": cy1 - cy0}


@mcp.tool()
def paste_region(
    session_id: str, x: int, y: int, layer_index: int | None = None, respect_alpha: bool = True,
) -> dict:
    """Paste the session's clipboard (see copy_region) with its top-left corner
    at (x, y). respect_alpha=True (default) only overwrites destination pixels
    where the clipboard content isn't fully transparent."""
    session = _get_session(session_id)
    if session.clipboard is None:
        raise ValueError("Clipboard is empty — call copy_region first")
    layer = _get_layer(session, layer_index)
    lh, lw = layer.pixels.shape[:2]
    ch, cw = session.clipboard.shape[:2]

    dst_x0, dst_y0 = max(0, x), max(0, y)
    dst_x1, dst_y1 = min(lw, x + cw), min(lh, y + ch)
    if dst_x1 <= dst_x0 or dst_y1 <= dst_y0:
        return {"pasted": False}
    src_x0, src_y0 = dst_x0 - x, dst_y0 - y
    src_x1, src_y1 = src_x0 + (dst_x1 - dst_x0), src_y0 + (dst_y1 - dst_y0)

    src = session.clipboard[src_y0:src_y1, src_x0:src_x1]
    dst = layer.pixels[dst_y0:dst_y1, dst_x0:dst_x1]
    if respect_alpha:
        mask = src[..., 3] > 0
        dst[mask] = src[mask]
    else:
        dst[:] = src
    return {"pasted": True, "x0": dst_x0, "y0": dst_y0, "x1": dst_x1, "y1": dst_y1}


@mcp.tool()
def flip_region(
    session_id: str, x0: int, y0: int, x1: int, y1: int, axis: str, layer_index: int | None = None,
) -> dict:
    """Flip a rectangular region in place. axis: "horizontal" or "vertical"."""
    session = _get_session(session_id)
    layer = _get_layer(session, layer_index)
    h, w = layer.pixels.shape[:2]
    cx0, cy0, cx1, cy1 = _clamp_rect(x0, y0, x1, y1, w, h)
    if axis not in ("horizontal", "vertical"):
        raise ValueError('axis must be "horizontal" or "vertical"')
    np_axis = 1 if axis == "horizontal" else 0
    region = layer.pixels[cy0:cy1, cx0:cx1]
    # np.flip returns a VIEW aliasing `region`'s memory; must copy before
    # mutating region below, or this reads back already-overwritten data.
    flipped = np.flip(region, axis=np_axis).copy()
    region[:] = flipped
    return {"x0": cx0, "y0": cy0, "x1": cx1, "y1": cy1}


@mcp.tool()
def rotate_region_90(
    session_id: str, x0: int, y0: int, x1: int, y1: int, clockwise: bool = True,
    layer_index: int | None = None,
) -> dict:
    """Rotate a rectangular region 90 degrees in place, re-centered on the same
    region's original center (its bounding box may change if it isn't square)."""
    session = _get_session(session_id)
    layer = _get_layer(session, layer_index)
    lh, lw = layer.pixels.shape[:2]
    cx0, cy0, cx1, cy1 = _clamp_rect(x0, y0, x1, y1, lw, lh)

    region = layer.pixels[cy0:cy1, cx0:cx1].copy()
    rotated = np.rot90(region, k=-1 if clockwise else 1)
    rh, rw = rotated.shape[:2]
    center_x, center_y = (cx0 + cx1) / 2.0, (cy0 + cy1) / 2.0
    nx0, ny0 = int(round(center_x - rw / 2.0)), int(round(center_y - rh / 2.0))

    dst_x0, dst_y0 = max(0, nx0), max(0, ny0)
    dst_x1, dst_y1 = min(lw, nx0 + rw), min(lh, ny0 + rh)
    if dst_x1 <= dst_x0 or dst_y1 <= dst_y0:
        return {"rotated": False}
    src_x0, src_y0 = dst_x0 - nx0, dst_y0 - ny0
    src_x1, src_y1 = src_x0 + (dst_x1 - dst_x0), src_y0 + (dst_y1 - dst_y0)

    layer.pixels[cy0:cy1, cx0:cx1] = 0
    layer.pixels[dst_y0:dst_y1, dst_x0:dst_x1] = rotated[src_y0:src_y1, src_x0:src_x1]
    return {"x0": dst_x0, "y0": dst_y0, "x1": dst_x1, "y1": dst_y1}


@mcp.tool()
def transform_region(
    session_id: str, x0: int, y0: int, x1: int, y1: int,
    scale_x: float = 1.0, scale_y: float = 1.0, rotate_degrees: float = 0.0,
    layer_index: int | None = None,
) -> dict:
    """Non-destructively scale and/or rotate a rectangular region by an
    arbitrary angle (RotSprite-style nearest-neighbor, better for pixel art
    than a naive rotation) and place the result back centered on the same
    spot. For a lossless 90-degree turn, prefer rotate_region_90 instead.
    """
    session = _get_session(session_id)
    layer = _get_layer(session, layer_index)
    lh, lw = layer.pixels.shape[:2]
    cx0, cy0, cx1, cy1 = _clamp_rect(x0, y0, x1, y1, lw, lh)

    mask = np.zeros((lh, lw), dtype=bool)
    mask[cy0:cy1, cx0:cx1] = True
    new_pixels, new_mask, (ox, oy) = transform_selection(layer.pixels, mask, scale_x, scale_y, rotate_degrees)
    layer.pixels[cy0:cy1, cx0:cx1] = 0

    nh, nw = new_pixels.shape[:2]
    dst_x0, dst_y0 = max(0, ox), max(0, oy)
    dst_x1, dst_y1 = min(lw, ox + nw), min(lh, oy + nh)
    if dst_x1 <= dst_x0 or dst_y1 <= dst_y0:
        return {"transformed": False}
    src_x0, src_y0 = dst_x0 - ox, dst_y0 - oy
    src_x1, src_y1 = src_x0 + (dst_x1 - dst_x0), src_y0 + (dst_y1 - dst_y0)

    src = new_pixels[src_y0:src_y1, src_x0:src_x1]
    dst = layer.pixels[dst_y0:dst_y1, dst_x0:dst_x1]
    src_mask = src[..., 3] > 0
    dst[src_mask] = src[src_mask]
    return {"x0": dst_x0, "y0": dst_y0, "x1": dst_x1, "y1": dst_y1}


# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------

@mcp.tool()
def get_palette(session_id: str) -> dict:
    """Get the session's current color palette."""
    session = _get_session(session_id)
    return {"name": session.palette.name, "colors": list(session.palette.colors)}


@mcp.tool()
def set_palette(session_id: str, colors: list[list[int]], name: str = "Custom") -> dict:
    """Replace the session's palette with the given list of [r,g,b] / [r,g,b,a] colors."""
    session = _get_session(session_id)
    session.palette = Palette(name=name, colors=[_color(c) for c in colors])
    return {"count": len(session.palette.colors)}


@mcp.tool()
def load_palette_file(session_id: str, path: str) -> dict:
    """Load a .gpl (GIMP) or .hex palette file into the session."""
    session = _get_session(session_id)
    session.palette = Palette.load_gpl(path) if path.lower().endswith(".gpl") else Palette.load_hex(path)
    return {"count": len(session.palette.colors)}


@mcp.tool()
def save_palette_file(session_id: str, path: str) -> dict:
    """Save the session's palette as a .gpl (GIMP) or .hex file."""
    session = _get_session(session_id)
    if path.lower().endswith(".hex"):
        session.palette.save_hex(path)
    else:
        session.palette.save_gpl(path)
    return {"path": path}


# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------

FILTER_NAMES = (
    "quantize", "outline", "ramp_shift", "autoclean", "drop_shadow",
    "blur", "sharpen", "brightness_contrast", "invert", "grayscale",
)


@mcp.tool()
def apply_filter(
    session_id: str,
    name: str,
    layer_index: int | None = None,
    color: list[int] | None = None,
    dithered: bool = False,
    shift: int = 1,
    offset: list[int] | None = None,
    radius: int = 1,
    amount: float = 1.0,
    brightness: int = 0,
    contrast: int = 0,
) -> dict:
    """Apply a pixel-art filter to a layer (default: the active layer).

    name must be one of: quantize, outline, ramp_shift, autoclean, drop_shadow,
    blur, sharpen, brightness_contrast, invert, grayscale. Only the parameters
    relevant to the chosen filter are used, by filter:
      quantize: (none, uses the session palette)
      outline: color=[r,g,b,a] (default black), dithered
      ramp_shift: shift (uses the session palette as the ramp)
      autoclean: (none)
      drop_shadow: offset=[dx,dy] (default [2,2]), color=[r,g,b,a] (default translucent black)
      blur: radius
      sharpen: amount, radius
      brightness_contrast: brightness (-255..255), contrast (-255..255)
      invert: (none)
      grayscale: (none)
    """
    session = _get_session(session_id)
    layer = _get_layer(session, layer_index)
    px = layer.pixels

    if name == "quantize":
        layer.pixels = flt.apply_palette_quantize(px, session.palette)
    elif name == "outline":
        outline_color = _color(color if color is not None else [0, 0, 0, 255])
        outline = flt.generate_outline(px, color=outline_color, dithered=dithered)
        merged = outline.copy()
        mask = px[..., 3] > 0
        merged[mask] = px[mask]
        layer.pixels = merged
    elif name == "ramp_shift":
        layer.pixels = flt.apply_color_ramp_shift(px, session.palette.colors, shift)
    elif name == "autoclean":
        layer.pixels = flt.auto_clean_double_pixels(px)
    elif name == "drop_shadow":
        shadow_offset = tuple(offset) if offset is not None else (2, 2)
        shadow_color = _color(color if color is not None else [0, 0, 0, 160])
        layer.pixels = flt.apply_drop_shadow(px, offset=shadow_offset, color=shadow_color)
    elif name == "blur":
        layer.pixels = flt.box_blur(px, radius=radius)
    elif name == "sharpen":
        layer.pixels = flt.sharpen(px, amount=amount, radius=radius)
    elif name == "brightness_contrast":
        layer.pixels = flt.adjust_brightness_contrast(px, brightness=brightness, contrast=contrast)
    elif name == "invert":
        layer.pixels = flt.invert_colors(px)
    elif name == "grayscale":
        layer.pixels = flt.grayscale(px)
    else:
        raise ValueError(f"Unknown filter {name!r}; must be one of {FILTER_NAMES}")
    return {"filter": name, "applied": True}


# ---------------------------------------------------------------------------
# Sprite sheet
# ---------------------------------------------------------------------------

@mcp.tool()
def set_sprite_sheet(
    session_id: str, frame_width: int, frame_height: int,
    columns: int, rows: int, padding: int = 0,
) -> dict:
    """Mark the session's canvas as a sprite sheet with a uniform frame grid."""
    session = _get_session(session_id)
    layout = SheetLayout.GRID if (columns > 1 and rows > 1) else (
        SheetLayout.LINEAR if rows == 1 and columns > 1 else
        SheetLayout.VERTICAL if columns == 1 and rows > 1 else SheetLayout.SINGLE
    )
    session.sprite_sheet = SpriteSheetInfo(frame_width, frame_height, columns, rows, padding, layout=layout)
    return session.sprite_sheet.to_dict()


@mcp.tool()
def get_sprite_sheet_info(session_id: str) -> dict:
    """Get the session's sprite sheet layout, or {"sprite_sheet": null} if none is set."""
    session = _get_session(session_id)
    return {"sprite_sheet": session.sprite_sheet.to_dict() if session.sprite_sheet else None}


@mcp.tool()
def get_frame_image(session_id: str, col: int, row: int) -> Image:
    """Get one sprite sheet frame as a PNG image, cropped from the composite."""
    session = _get_session(session_id)
    if session.sprite_sheet is None:
        raise ValueError("This session has no sprite sheet layout set")
    x0, y0, x1, y1 = session.sprite_sheet.frame_rect(col, row)
    crop = session.layer_stack.composite()[y0:y1, x0:x1]
    return Image(data=_png_bytes(crop), format="png")


# ---------------------------------------------------------------------------
# Vision: let the agent see its work
# ---------------------------------------------------------------------------

@mcp.tool()
def get_image(session_id: str, layer_index: int | None = None) -> Image:
    """Get the current canvas as a PNG image: the flattened composite by
    default, or a single layer's pixels if layer_index is given. Call this
    after making edits to see the actual result before deciding what to do next.
    """
    session = _get_session(session_id)
    pixels = session.layer_stack.composite() if layer_index is None else _get_layer(session, layer_index).pixels
    return Image(data=_png_bytes(pixels), format="png")


@mcp.tool()
def get_image_base64(session_id: str, layer_index: int | None = None) -> dict:
    """Same as get_image, but returns a base64-encoded PNG string instead of a
    native image attachment, for clients that can't display attachments."""
    session = _get_session(session_id)
    pixels = session.layer_stack.composite() if layer_index is None else _get_layer(session, layer_index).pixels
    return {"png_base64": base64.b64encode(_png_bytes(pixels)).decode("ascii")}


def _upscale_nearest(pixels: np.ndarray, scale: int) -> np.ndarray:
    return np.repeat(np.repeat(pixels, scale, axis=0), scale, axis=1)


def _draw_grid_lines(pixels: np.ndarray, cell: int, color: tuple[int, int, int, int]) -> None:
    """Draw grid lines onto an already-upscaled array in place, `cell` pixels apart."""
    h, w = pixels.shape[:2]
    pixels[0:h:cell, :] = color
    pixels[:, 0:w:cell] = color


@mcp.tool()
def get_screenshot(
    session_id: str,
    scale: int = 8,
    show_pixel_grid: bool = False,
    show_frame_markers: bool = True,
    layer_index: int | None = None,
) -> Image:
    """Get a nearest-neighbor upscaled "screenshot" of the canvas — much easier
    to actually see than raw get_image for small canvases, since e.g. a 16x16
    sprite renders as a 16x16 postage stamp otherwise. show_pixel_grid draws
    1px-cell grid lines; show_frame_markers (default on) outlines sprite sheet
    frame boundaries in a bright color, if a sprite sheet layout is set.
    Always look at this (or get_image) after making edits.
    """
    session = _get_session(session_id)
    pixels = session.layer_stack.composite() if layer_index is None else _get_layer(session, layer_index).pixels
    scale = max(1, min(32, scale))
    big = _upscale_nearest(pixels, scale)

    if show_pixel_grid and scale >= 4:
        _draw_grid_lines(big, scale, (255, 255, 255, 60))

    if show_frame_markers and session.sprite_sheet is not None:
        sheet = session.sprite_sheet
        marker = (255, 140, 40, 255)
        for row in range(sheet.rows):
            for col in range(sheet.columns):
                x0, y0, x1, y1 = sheet.frame_rect(col, row)
                sx0, sy0, sx1, sy1 = x0 * scale, y0 * scale, x1 * scale, y1 * scale
                big[sy0:sy1, sx0:sx0 + 1] = marker
                big[sy0:sy1, sx1 - 1:sx1] = marker
                big[sy0:sy0 + 1, sx0:sx1] = marker
                big[sy1 - 1:sy1, sx0:sx1] = marker

    return Image(data=_png_bytes(big), format="png")


@mcp.tool()
def validate_render(session_id: str, layer_index: int | None = None) -> dict:
    """Get diagnostic stats about the current render, to sanity-check work
    without relying solely on visual inspection: whether it's empty, the
    tight bounding box of non-transparent content, how many unique colors are
    used, what fraction of pixels match the session palette exactly (only
    meaningful if you've set one), and — if a sprite sheet is defined — the
    same breakdown per frame, so you can catch an accidentally-blank or
    off-palette frame in an animation.
    """
    session = _get_session(session_id)
    pixels = session.layer_stack.composite() if layer_index is None else _get_layer(session, layer_index).pixels

    def stats_for(region: np.ndarray) -> dict:
        alpha = region[..., 3] > 0
        non_transparent = int(alpha.sum())
        total = alpha.size
        if non_transparent == 0:
            bbox = None
        else:
            ys, xs = np.where(alpha)
            bbox = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
        flat = region.reshape(-1, 4)
        unique_colors = int(np.unique(flat[flat[:, 3] > 0], axis=0).shape[0]) if non_transparent else 0
        result = {
            "width": region.shape[1], "height": region.shape[0],
            "is_empty": non_transparent == 0,
            "non_transparent_pixels": non_transparent,
            "transparent_pixels": total - non_transparent,
            "bounding_box": bbox,
            "unique_colors": unique_colors,
        }
        if session.palette.colors and non_transparent:
            palette_arr = session.palette.as_array()[:, :3].astype(np.int32)
            opaque = flat[flat[:, 3] > 0][:, :3].astype(np.int32)
            matches = np.any(np.all(opaque[:, None, :] == palette_arr[None, :, :], axis=-1), axis=-1)
            result["palette_compliance"] = float(matches.mean())
        return result

    info: dict = {"overall": stats_for(pixels)}
    if session.sprite_sheet is not None:
        sheet = session.sprite_sheet
        frames = []
        for row in range(sheet.rows):
            for col in range(sheet.columns):
                x0, y0, x1, y1 = sheet.frame_rect(col, row)
                frame_stats = stats_for(pixels[y0:y1, x0:x1])
                frame_stats["col"], frame_stats["row"] = col, row
                frames.append(frame_stats)
        info["frames"] = frames
    return info


if __name__ == "__main__":
    mcp.run()
