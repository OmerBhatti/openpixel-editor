"""Shared QImage <-> numpy conversion and PNG/JPEG/WebP file I/O, including
embedding/reading a sprite sheet's exact frame layout as PNG metadata.

Used by both the desktop app (main.py) and the headless MCP server
(mcp_server.py) so this logic has a single source of truth instead of two
copies drifting apart.
"""
from __future__ import annotations

import json

import numpy as np
from PySide6.QtCore import QBuffer, QIODevice
from PySide6.QtGui import QImage

from sprite_sheet import SpriteSheetInfo

SPRITE_SHEET_METADATA_KEY = "openpixel:sprite_sheet"

IMPORT_IMAGE_FILTER = "Images (*.png *.jpg *.jpeg *.webp);;PNG (*.png);;JPEG (*.jpg *.jpeg);;WebP (*.webp)"


def pixels_to_qimage(pixels: np.ndarray) -> QImage:
    h, w = pixels.shape[:2]
    return QImage(np.ascontiguousarray(pixels).tobytes(), w, h, w * 4, QImage.Format.Format_RGBA8888)


def qimage_to_pixels(image: QImage) -> np.ndarray:
    image = image.convertToFormat(QImage.Format.Format_RGBA8888)
    w, h = image.width(), image.height()
    return np.frombuffer(image.bits(), dtype=np.uint8, count=h * w * 4).reshape((h, w, 4)).copy()


def png_bytes(pixels: np.ndarray) -> bytes:
    """Encode an RGBA array as in-memory PNG bytes (no sprite-sheet metadata;
    use save_rgba_png for that, since a text chunk only makes sense on disk)."""
    image = pixels_to_qimage(pixels)
    buf = QBuffer()
    buf.open(QIODevice.OpenModeFlag.WriteOnly)
    image.save(buf, "PNG")
    return bytes(buf.data())


def save_rgba_png(pixels: np.ndarray, path: str, sprite_sheet: SpriteSheetInfo | None = None) -> None:
    """Save an RGBA array as a PNG file, optionally embedding the sprite sheet's
    exact frame layout as a text chunk so re-importing this file later can read
    it back directly instead of re-guessing via gutter-detection.
    """
    image = pixels_to_qimage(pixels)
    if sprite_sheet is not None:
        image.setText(SPRITE_SHEET_METADATA_KEY, json.dumps(sprite_sheet.to_dict()))
    if not image.save(path, "PNG"):
        raise ValueError(f"Failed to write PNG: {path}")


def load_rgba_image(path: str) -> np.ndarray | None:
    """Load any Qt-supported raster format (PNG, JPEG, WebP, ...) as RGBA uint8.
    Returns None if the file couldn't be decoded.
    """
    raw = QImage(path)
    if raw.isNull():
        return None
    return qimage_to_pixels(raw)


def read_sprite_sheet_metadata(path: str) -> SpriteSheetInfo | None:
    """Read back sprite-sheet layout embedded by save_rgba_png, if any. Only PNG
    reliably carries this (JPEG/WebP re-encoding could strip or corrupt it)."""
    if not path.lower().endswith(".png"):
        return None
    raw = QImage(path)
    if raw.isNull():
        return None
    text = raw.text(SPRITE_SHEET_METADATA_KEY)
    if not text:
        return None
    try:
        return SpriteSheetInfo.from_dict(json.loads(text))
    except (ValueError, KeyError, TypeError):
        return None
