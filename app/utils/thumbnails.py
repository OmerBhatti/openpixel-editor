"""Small QPixmap thumbnail renderers shared by the dock panels: brush coverage
matrices, raw matrices (used live in the Brush Matrix Editor preview), and
layer contents composited over white (matching the canvas's default
"white means transparent" background convention).
"""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPainter, QPixmap

from brushes import Brush


def render_matrix_thumbnail(matrix: np.ndarray, size: int = 26, on_dark: bool = True) -> QPixmap:
    """Render a coverage matrix (0-255 alpha) as a thumbnail, light marks on a
    transparent background so it reads against either a dark or white panel.
    """
    h, w = matrix.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[..., 0:3] = 235 if on_dark else 40
    rgba[..., 3] = matrix
    image = QImage(np.ascontiguousarray(rgba).tobytes(), w, h, w * 4, QImage.Format.Format_RGBA8888)
    return QPixmap.fromImage(image).scaled(
        size, size, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.FastTransformation
    )


def render_brush_thumbnail(brush: Brush, size: int = 26) -> QPixmap:
    return render_matrix_thumbnail(brush.matrix, size)


def render_layer_thumbnail(pixels: np.ndarray, size: int = 28) -> QPixmap:
    """Render a layer's RGBA buffer composited over white (white == transparent,
    matching the canvas's default background) as a small thumbnail."""
    alpha = pixels[..., 3:4].astype(np.float32) / 255.0
    rgb = pixels[..., :3].astype(np.float32)
    composited = rgb * alpha + 255.0 * (1.0 - alpha)
    h, w = pixels.shape[:2]
    out = np.dstack([composited.astype(np.uint8), np.full((h, w, 1), 255, dtype=np.uint8)])
    image = QImage(np.ascontiguousarray(out).tobytes(), w, h, w * 4, QImage.Format.Format_RGBA8888)
    pixmap = QPixmap.fromImage(image).scaled(
        size, size, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.FastTransformation
    )
    framed = QPixmap(size, size)
    framed.fill(Qt.GlobalColor.transparent)
    painter = QPainter(framed)
    x = (size - pixmap.width()) // 2
    y = (size - pixmap.height()) // 2
    painter.drawPixmap(x, y, pixmap)
    painter.end()
    return framed
