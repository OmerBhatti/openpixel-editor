"""Layer stack data model: RGBA uint8 buffers, blend modes, opacity, visibility."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np


class BlendMode(Enum):
    NORMAL = "Normal"
    MULTIPLY = "Multiply"
    SCREEN = "Screen"


@dataclass
class Layer:
    name: str
    pixels: np.ndarray  # (H, W, 4) uint8 RGBA
    opacity: float = 1.0
    visible: bool = True
    blend_mode: BlendMode = BlendMode.NORMAL
    locked: bool = False

    @classmethod
    def blank(cls, name: str, width: int, height: int) -> "Layer":
        return cls(name=name, pixels=np.zeros((height, width, 4), dtype=np.uint8))


def _blend_pair(base: np.ndarray, top: np.ndarray, mode: BlendMode, opacity: float) -> np.ndarray:
    """Composite `top` over `base`, both (H,W,4) float32 in [0,1]. Returns straight-alpha RGBA float."""
    base_rgb, base_a = base[..., :3], base[..., 3:4]
    top_rgb, top_a = top[..., :3], top[..., 3:4] * opacity

    if mode == BlendMode.MULTIPLY:
        blended_rgb = base_rgb * top_rgb
    elif mode == BlendMode.SCREEN:
        blended_rgb = 1.0 - (1.0 - base_rgb) * (1.0 - top_rgb)
    else:
        blended_rgb = top_rgb

    out_a = top_a + base_a * (1.0 - top_a)
    safe_a = np.where(out_a == 0, 1.0, out_a)
    mixed_rgb = (blended_rgb * top_a + base_rgb * base_a * (1.0 - top_a)) / safe_a
    out = np.concatenate([mixed_rgb, out_a], axis=-1)
    return out


class LayerStack:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.layers: list[Layer] = [Layer.blank("Layer 1", width, height)]
        self.active_index: int = 0

    @property
    def active(self) -> Layer:
        return self.layers[self.active_index]

    def add_layer(self, name: str | None = None, at: int | None = None) -> Layer:
        layer = Layer.blank(name or f"Layer {len(self.layers) + 1}", self.width, self.height)
        idx = at if at is not None else self.active_index + 1
        self.layers.insert(idx, layer)
        self.active_index = idx
        return layer

    def remove_layer(self, index: int) -> None:
        if len(self.layers) <= 1:
            return
        del self.layers[index]
        self.active_index = min(self.active_index, len(self.layers) - 1)

    def move_layer(self, src: int, dst: int) -> None:
        layer = self.layers.pop(src)
        self.layers.insert(dst, layer)
        self.active_index = dst

    def composite(self) -> np.ndarray:
        """Flatten the visible layer stack into a single (H, W, 4) uint8 RGBA image."""
        result = np.zeros((self.height, self.width, 4), dtype=np.float32)
        for layer in self.layers:
            if not layer.visible or layer.opacity <= 0:
                continue
            top = layer.pixels.astype(np.float32) / 255.0
            result = _blend_pair(result, top, layer.blend_mode, layer.opacity)
        return np.clip(result * 255.0 + 0.5, 0, 255).astype(np.uint8)

    def resize_canvas(self, width: int, height: int) -> None:
        for layer in self.layers:
            new_pixels = np.zeros((height, width, 4), dtype=np.uint8)
            h = min(height, layer.pixels.shape[0])
            w = min(width, layer.pixels.shape[1])
            new_pixels[:h, :w] = layer.pixels[:h, :w]
            layer.pixels = new_pixels
        self.width, self.height = width, height
