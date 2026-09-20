"""Palette management: indexed color palettes, .gpl/.hex import/export, nearest-color matching."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Palette:
    name: str = "Untitled Palette"
    colors: list[tuple[int, int, int, int]] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.colors)

    def add(self, rgba: tuple[int, int, int, int]) -> int:
        """Append `rgba` unless it's already present; returns the color's index either way."""
        if rgba in self.colors:
            return self.colors.index(rgba)
        self.colors.append(rgba)
        return len(self.colors) - 1

    def remove(self, index: int) -> None:
        if 0 <= index < len(self.colors):
            del self.colors[index]

    def as_array(self) -> np.ndarray:
        if not self.colors:
            return np.zeros((0, 4), dtype=np.uint8)
        return np.array(self.colors, dtype=np.uint8)

    def nearest_index(self, rgba: tuple[int, int, int, int]) -> int:
        if not self.colors:
            return -1
        arr = self.as_array().astype(np.int32)
        target = np.array(rgba[:3], dtype=np.int32)
        diff = arr[:, :3] - target
        dist = np.einsum("ij,ij->i", diff, diff)
        return int(np.argmin(dist))

    def nearest_color(self, rgba: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
        idx = self.nearest_index(rgba)
        if idx < 0:
            return rgba
        return self.colors[idx]

    # ---- Import / Export ----

    def save_gpl(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            f.write("GIMP Palette\n")
            f.write(f"Name: {self.name}\n")
            f.write("Columns: 16\n#\n")
            for i, (r, g, b, a) in enumerate(self.colors):
                f.write(f"{r:3d} {g:3d} {b:3d}\tindex{i}\n")

    @classmethod
    def load_gpl(cls, path: str) -> "Palette":
        pal = cls(name="Imported Palette")
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        for line in lines:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("GIMP Palette"):
                continue
            if line.startswith("Name:"):
                pal.name = line.split(":", 1)[1].strip()
                continue
            if line.startswith("Columns:"):
                continue
            parts = line.split()
            if len(parts) >= 3:
                try:
                    r, g, b = int(parts[0]), int(parts[1]), int(parts[2])
                    pal.add((r, g, b, 255))
                except ValueError:
                    continue
        return pal

    def save_hex(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            for r, g, b, a in self.colors:
                f.write(f"{r:02X}{g:02X}{b:02X}\n")

    @classmethod
    def load_hex(cls, path: str) -> "Palette":
        pal = cls(name="Imported Palette")
        hex_re = re.compile(r"^#?([0-9A-Fa-f]{6})$")
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                m = hex_re.match(line)
                if m:
                    h = m.group(1)
                    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
                    pal.add((r, g, b, 255))
        return pal

    @classmethod
    def default(cls) -> "Palette":
        """A small default 16-color palette (DB16-ish)."""
        base = [
            (20, 12, 28, 255), (68, 36, 52, 255), (48, 52, 109, 255), (78, 74, 78, 255),
            (133, 76, 48, 255), (52, 101, 36, 255), (208, 70, 72, 255), (117, 113, 97, 255),
            (89, 125, 206, 255), (210, 125, 44, 255), (133, 149, 161, 255), (109, 170, 44,
            255), (210, 170, 153, 255), (109, 194, 202, 255), (218, 212, 94, 255), (222, 238,
            214, 255),
        ]
        return cls(name="Default 16", colors=base)


def remap_indices(index_buffer: np.ndarray, mapping: dict[int, int]) -> np.ndarray:
    """Remap an indexed-color buffer (H,W int) live according to `mapping` (old_idx -> new_idx)."""
    out = index_buffer.copy()
    for old, new in mapping.items():
        out[index_buffer == old] = new
    return out


def quantize_to_palette(rgba_image: np.ndarray, palette: Palette) -> np.ndarray:
    """Map an RGBA image (H,W,4 uint8) to the nearest colors in `palette`."""
    if len(palette) == 0:
        return rgba_image.copy()
    h, w = rgba_image.shape[:2]
    flat = rgba_image.reshape(-1, 4).astype(np.int32)
    pal = palette.as_array().astype(np.int32)
    # Compute squared distance of every pixel to every palette color (vectorized).
    diff = flat[:, None, :3] - pal[None, :, :3]
    dist = np.einsum("ijk,ijk->ij", diff, diff)
    nearest = np.argmin(dist, axis=1)
    out_rgb = pal[nearest, :3]
    out = np.concatenate([out_rgb, flat[:, 3:4]], axis=1).astype(np.uint8)
    return out.reshape(h, w, 4)
