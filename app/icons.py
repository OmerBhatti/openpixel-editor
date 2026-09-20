"""SVG icon loading helper. Icons live as flat files in icons/ and are loaded as QIcon."""
from __future__ import annotations

import os

from PySide6.QtCore import QSize
from PySide6.QtGui import QIcon

_ICON_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "icons")
_cache: dict[str, QIcon] = {}


def icon(name: str) -> QIcon:
    """Load (and cache) an SVG icon by base name, e.g. icon('pencil') -> icons/pencil.svg."""
    if name not in _cache:
        path = os.path.join(_ICON_DIR, f"{name}.svg")
        _cache[name] = QIcon(path)
    return _cache[name]


ICON_SIZE = QSize(20, 20)
