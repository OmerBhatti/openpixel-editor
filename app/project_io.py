"""OpenPixel project file (.opxproj) save/load: a zip archive containing a JSON
manifest (canvas size, per-layer metadata) plus one lossless PNG per layer.
"""
from __future__ import annotations

import json
import zipfile

import numpy as np
from PySide6.QtCore import QBuffer, QIODevice
from PySide6.QtGui import QImage

from layers import Layer, LayerStack, BlendMode
from palette import Palette
from sprite_sheet import SpriteSheetInfo

MANIFEST_NAME = "manifest.json"
FORMAT_VERSION = 1


def _encode_png(pixels: np.ndarray) -> bytes:
    h, w = pixels.shape[:2]
    image = QImage(np.ascontiguousarray(pixels).tobytes(), w, h, w * 4, QImage.Format.Format_RGBA8888)
    qbuf = QBuffer()
    qbuf.open(QIODevice.OpenModeFlag.WriteOnly)
    image.save(qbuf, "PNG")
    return bytes(qbuf.data())


def _decode_png(data: bytes) -> np.ndarray:
    image = QImage.fromData(data, "PNG").convertToFormat(QImage.Format.Format_RGBA8888)
    w, h = image.width(), image.height()
    ptr = image.bits()
    return np.frombuffer(ptr, dtype=np.uint8, count=h * w * 4).reshape((h, w, 4)).copy()


def save_project(
    path: str,
    layer_stack: LayerStack,
    palette: Palette,
    project_name: str = "Untitled",
    sprite_sheet: SpriteSheetInfo | None = None,
) -> None:
    manifest = {
        "format_version": FORMAT_VERSION,
        "project_name": project_name,
        "width": layer_stack.width,
        "height": layer_stack.height,
        "active_index": layer_stack.active_index,
        "layers": [
            {
                "name": layer.name,
                "opacity": layer.opacity,
                "visible": layer.visible,
                "blend_mode": layer.blend_mode.value,
                "locked": layer.locked,
                "file": f"layer_{i}.png",
            }
            for i, layer in enumerate(layer_stack.layers)
        ],
        "palette_name": palette.name,
        "palette_colors": palette.colors,
        "sprite_sheet": sprite_sheet.to_dict() if sprite_sheet else None,
    }

    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(MANIFEST_NAME, json.dumps(manifest, indent=2))
        for i, layer in enumerate(layer_stack.layers):
            zf.writestr(f"layer_{i}.png", _encode_png(layer.pixels))


def load_project(path: str) -> tuple[LayerStack, Palette, str, SpriteSheetInfo | None]:
    with zipfile.ZipFile(path, "r") as zf:
        manifest = json.loads(zf.read(MANIFEST_NAME).decode("utf-8"))

        stack = LayerStack(manifest["width"], manifest["height"])
        stack.layers = []
        for layer_meta in manifest["layers"]:
            pixels = _decode_png(zf.read(layer_meta["file"]))
            layer = Layer(
                name=layer_meta["name"],
                pixels=pixels,
                opacity=layer_meta["opacity"],
                visible=layer_meta["visible"],
                blend_mode=BlendMode(layer_meta["blend_mode"]),
                locked=layer_meta.get("locked", False),
            )
            stack.layers.append(layer)
        stack.active_index = min(manifest.get("active_index", 0), len(stack.layers) - 1)

        palette = Palette(
            name=manifest.get("palette_name", "Untitled Palette"),
            colors=[tuple(c) for c in manifest.get("palette_colors", [])],
        )
        project_name = manifest.get("project_name", "Untitled")
        sheet_data = manifest.get("sprite_sheet")
        sprite_sheet = SpriteSheetInfo.from_dict(sheet_data) if sheet_data else None

    return stack, palette, project_name, sprite_sheet
