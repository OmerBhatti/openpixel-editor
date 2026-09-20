"""OpenPixel: a production-grade open-source 2D pixel art editor.

Entry point wiring together the canvas, tools, layers, palette, brushes, and filters
into a dark-themed PySide6 desktop application.
"""
from __future__ import annotations

import os
import sys

import numpy as np
from PySide6.QtCore import Qt, QSize, Signal
from PySide6.QtGui import (
    QAction, QColor, QIcon, QPainter, QPixmap, QUndoCommand, QUndoStack, QImage,
)
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QDockWidget, QListWidget, QListWidgetItem, QToolBar,
    QVBoxLayout, QHBoxLayout, QWidget, QLabel, QSlider, QPushButton, QColorDialog,
    QFileDialog, QSpinBox, QCheckBox, QDoubleSpinBox, QDialog, QGridLayout, QComboBox,
    QMessageBox, QButtonGroup, QToolButton, QStyle, QInputDialog,
)

from canvas import PixelCanvas
from layers import LayerStack, BlendMode, Layer
from palette import Palette
from brushes import Brush, BrushManager, make_dither_brush
from tools import ToolKind, Selection
import filters as flt
from transform import transform_selection
from icons import icon, ICON_SIZE
from project_io import save_project, load_project

DARK_STYLESHEET = """
QWidget { background-color: #2b2b2e; color: #d8d8d8; font-size: 12px; }
QMainWindow::separator { background: #1c1c1e; width: 2px; height: 2px; }
QDockWidget::title { background: #232326; padding: 4px; }
QToolBar { background: #232326; border: none; spacing: 4px; }
QListWidget { background: #232326; border: 1px solid #1c1c1e; }
QPushButton { background: #3a3a3e; border: 1px solid #1c1c1e; padding: 4px 8px; border-radius: 2px; }
QPushButton:hover { background: #46464a; }
QPushButton:checked { background: #5b7fbd; }
QSlider::groove:horizontal { background: #1c1c1e; height: 4px; }
QSlider::handle:horizontal { background: #5b7fbd; width: 10px; margin: -4px 0; border-radius: 2px; }
QComboBox, QSpinBox, QDoubleSpinBox { background: #3a3a3e; border: 1px solid #1c1c1e; padding: 2px; }
QMenuBar { background: #232326; }
QMenuBar::item:selected { background: #46464a; }
QMenu { background: #232326; border: 1px solid #1c1c1e; }
QMenu::item:selected { background: #5b7fbd; }
"""


# ---------------------------------------------------------------------------
# Undo/Redo
# ---------------------------------------------------------------------------

class PaintCommand(QUndoCommand):
    """Undoable stroke: stores full before/after pixel buffers for a single layer."""

    def __init__(self, layer_stack: LayerStack, layer_index: int, before: np.ndarray,
                 after: np.ndarray, canvas: PixelCanvas, text: str = "Paint"):
        super().__init__(text)
        self.layer_stack = layer_stack
        self.layer_index = layer_index
        self.before = before
        self.after = after
        self.canvas = canvas

    def redo(self) -> None:
        self.layer_stack.layers[self.layer_index].pixels = self.after.copy()
        self.canvas.mark_dirty()

    def undo(self) -> None:
        self.layer_stack.layers[self.layer_index].pixels = self.before.copy()
        self.canvas.mark_dirty()


# ---------------------------------------------------------------------------
# Brush Matrix Editor Dialog
# ---------------------------------------------------------------------------

class BrushMatrixEditor(QDialog):
    """Editable pixel grid dialog for authoring custom stamp/dither brushes."""

    def __init__(self, size: int = 8, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Brush Matrix Editor")
        self.size = size
        self.matrix = np.zeros((size, size), dtype=np.uint8)

        layout = QVBoxLayout(self)
        self.name_box = QComboBox(self)
        self.name_box.setEditable(True)
        self.name_box.addItem("Custom Brush")
        layout.addWidget(self.name_box)

        top_row = QHBoxLayout()

        self.grid = QGridLayout()
        self.grid.setSpacing(1)
        self.buttons: list[list[QToolButton]] = []
        for y in range(size):
            row = []
            for x in range(size):
                btn = QToolButton()
                btn.setCheckable(True)
                btn.setFixedSize(28, 28)
                btn.clicked.connect(lambda checked, xx=x, yy=y: self._toggle(xx, yy, checked))
                self.grid.addWidget(btn, y, x)
                row.append(btn)
            self.buttons.append(row)
        top_row.addLayout(self.grid)

        preview_col = QVBoxLayout()
        preview_col.addWidget(QLabel("Preview:"))
        self.preview_label = QLabel()
        self.preview_label.setFixedSize(72, 72)
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setStyleSheet("background-color: #232326; border: 1px solid #1c1c1e;")
        preview_col.addWidget(self.preview_label)
        preview_col.addStretch(1)
        top_row.addLayout(preview_col)

        layout.addLayout(top_row)

        btn_row = QHBoxLayout()
        save_btn = QPushButton("Save as .pxbrush")
        save_btn.clicked.connect(self._save)
        ok_btn = QPushButton("Use Brush")
        ok_btn.clicked.connect(self.accept)
        btn_row.addWidget(save_btn)
        btn_row.addWidget(ok_btn)
        layout.addLayout(btn_row)

        self._update_preview()

    def _toggle(self, x: int, y: int, checked: bool) -> None:
        self.matrix[y, x] = 255 if checked else 0
        self._update_preview()

    def _update_preview(self) -> None:
        pixmap = _render_matrix_thumbnail(self.matrix, size=64, on_dark=True)
        self.preview_label.setPixmap(pixmap)

    def _save(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Save Brush", "", "Pixel Brush (*.pxbrush)")
        if path:
            brush = Brush(name=self.name_box.currentText(), matrix=self.matrix.copy())
            brush.save(path)

    def result_brush(self) -> Brush:
        return Brush(name=self.name_box.currentText(), matrix=self.matrix.copy())


# ---------------------------------------------------------------------------
# Sprite Sheet Exporter Dialog
# ---------------------------------------------------------------------------

class SpriteSheetExportDialog(QDialog):
    def __init__(self, layer_stack: LayerStack, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Sprite Sheet Exporter")
        self.layer_stack = layer_stack
        layout = QGridLayout(self)

        layout.addWidget(QLabel("Columns:"), 0, 0)
        self.cols = QSpinBox()
        self.cols.setRange(1, 64)
        self.cols.setValue(max(1, len(layer_stack.layers)))
        layout.addWidget(self.cols, 0, 1)

        layout.addWidget(QLabel("Rows:"), 1, 0)
        self.rows = QSpinBox()
        self.rows.setRange(1, 64)
        self.rows.setValue(1)
        layout.addWidget(self.rows, 1, 1)

        layout.addWidget(QLabel("Frames come from:"), 2, 0)
        self.source = QComboBox()
        self.source.addItems(["Each Layer", "Single Composite"])
        layout.addWidget(self.source, 2, 1)

        export_btn = QPushButton("Export PNG Sprite Sheet")
        export_btn.clicked.connect(self._export)
        layout.addWidget(export_btn, 3, 0, 1, 2)

    def _export(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Export Sprite Sheet", "", "PNG Image (*.png)")
        if not path:
            return
        w, h = self.layer_stack.width, self.layer_stack.height
        cols, rows = self.cols.value(), self.rows.value()

        if self.source.currentText() == "Each Layer":
            frames = [layer.pixels for layer in self.layer_stack.layers]
        else:
            frames = [self.layer_stack.composite()]

        sheet = np.zeros((h * rows, w * cols, 4), dtype=np.uint8)
        for i, frame in enumerate(frames):
            if i >= cols * rows:
                break
            r, c = divmod(i, cols)
            sheet[r * h:(r + 1) * h, c * w:(c + 1) * w] = frame

        save_rgba_png(sheet, path)
        QMessageBox.information(self, "Export Complete", f"Sprite sheet exported to {path}")


NEW_DOC_STYLESHEET = """
QLabel { color: #b8b8b8; }
QLineEdit, QSpinBox, QComboBox {
    background: #1e1e20; border: 1px solid #444448; border-radius: 3px;
    padding: 5px 6px; color: #e8e8e8;
}
QPushButton#createBtn {
    background: #565a60; border: none; border-radius: 3px; padding: 8px; font-weight: 600;
}
QPushButton#createBtn:hover { background: #6a6e74; }
QCheckBox { color: #b8b8b8; }
"""


class NewDocumentDialog(QDialog):
    """Photoshop-style "New Document" dialog: a single stacked form (Name, Width/Height
    with a unit swap, DPI, Background + swatch, Mode + bit depth, Profile).
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("New Document")
        self.setFixedWidth(360)
        self.setStyleSheet(NEW_DOC_STYLESHEET)

        root = QVBoxLayout(self)
        root.setSpacing(14)

        def section(label_text: str) -> QLabel:
            lbl = QLabel(label_text)
            root.addWidget(lbl)
            return lbl

        section("Name")
        self.name_edit = QComboBox()
        self.name_edit.setEditable(True)
        self.name_edit.setCurrentText("New Project")
        root.addWidget(self.name_edit)

        size_row = QHBoxLayout()
        size_row.setSpacing(10)

        width_col = QVBoxLayout()
        width_col.addWidget(QLabel("Width"))
        self.width_box = QSpinBox()
        self.width_box.setRange(1, 8192)
        self.width_box.setValue(16)
        width_col.addWidget(self.width_box)
        size_row.addLayout(width_col)

        height_col = QVBoxLayout()
        height_col.addWidget(QLabel("Height"))
        self.height_box = QSpinBox()
        self.height_box.setRange(1, 8192)
        self.height_box.setValue(16)
        height_col.addWidget(self.height_box)
        size_row.addLayout(height_col)

        swap_col = QVBoxLayout()
        swap_col.addWidget(QLabel(""))
        self.swap_btn = QToolButton()
        self.swap_btn.setIcon(icon("swap_colors"))
        self.swap_btn.setIconSize(QSize(14, 14))
        self.swap_btn.setToolTip("Swap width and height")
        self.swap_btn.clicked.connect(self._swap_dimensions)
        swap_col.addWidget(self.swap_btn)
        size_row.addLayout(swap_col)

        unit_col = QVBoxLayout()
        unit_col.addWidget(QLabel(""))
        self.unit_combo = QComboBox()
        self.unit_combo.addItems(["Pixels", "Inches", "Centimeters"])
        unit_col.addWidget(self.unit_combo)
        size_row.addLayout(unit_col, 1)

        root.addLayout(size_row)

        section("DPI")
        dpi_row = QHBoxLayout()
        self.dpi_box = QSpinBox()
        self.dpi_box.setRange(1, 1200)
        self.dpi_box.setValue(72)
        dpi_row.addWidget(self.dpi_box)
        self.dpi_unit_combo = QComboBox()
        self.dpi_unit_combo.addItems(["Pixels / Inch"])
        dpi_row.addWidget(self.dpi_unit_combo, 1)
        root.addLayout(dpi_row)

        section("Background")
        bg_row = QHBoxLayout()
        self.background_combo = QComboBox()
        self.background_combo.addItems(["Transparent", "White", "Black"])
        self.background_combo.currentTextChanged.connect(self._sync_background_swatch)
        bg_row.addWidget(self.background_combo, 1)

        self.bg_swatch = QLabel()
        self.bg_swatch.setFixedSize(24, 24)
        self.bg_swatch.setStyleSheet("background: transparent; border: 1px dashed #888; border-radius: 2px;")
        bg_row.addWidget(self.bg_swatch)
        root.addLayout(bg_row)

        mode_row = QHBoxLayout()
        mode_col = QVBoxLayout()
        mode_col.addWidget(QLabel("Mode"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["RGB", "Indexed"])
        mode_col.addWidget(self.mode_combo)
        mode_row.addLayout(mode_col, 1)

        depth_col = QVBoxLayout()
        depth_col.addWidget(QLabel(""))
        self.depth_combo = QComboBox()
        self.depth_combo.addItems(["8 bit"])
        depth_col.addWidget(self.depth_combo)
        mode_row.addLayout(depth_col)
        root.addLayout(mode_row)

        section("Profile")
        self.profile_combo = QComboBox()
        self.profile_combo.addItems(["sRGB", "None"])
        root.addWidget(self.profile_combo)

        create_btn = QPushButton("Create")
        create_btn.setObjectName("createBtn")
        create_btn.setDefault(True)
        create_btn.setMinimumHeight(34)
        create_btn.clicked.connect(self.accept)
        root.addWidget(create_btn)

        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        root.addWidget(cancel_btn)

    def _swap_dimensions(self) -> None:
        w, h = self.width_box.value(), self.height_box.value()
        self.width_box.setValue(h)
        self.height_box.setValue(w)

    def _sync_background_swatch(self, text: str) -> None:
        color = {"White": "#ffffff", "Black": "#000000", "Transparent": "transparent"}[text]
        border = "1px dashed #888" if text == "Transparent" else "1px solid #444448"
        self.bg_swatch.setStyleSheet(f"background: {color}; border: {border}; border-radius: 2px;")

    def result_values(self) -> tuple[int, int, tuple[int, int, int, int], str]:
        w, h = self.width_box.value(), self.height_box.value()
        bg_text = self.background_combo.currentText()
        background = {
            "White": (255, 255, 255, 255),
            "Black": (0, 0, 0, 255),
            "Transparent": (0, 0, 0, 0),
        }[bg_text]
        name = self.name_edit.currentText().strip() or "New Project"
        return w, h, background, name


def save_rgba_png(array: np.ndarray, path: str) -> None:
    h, w = array.shape[:2]
    image = QImage(array.tobytes(), w, h, w * 4, QImage.Format.Format_RGBA8888)
    image.save(path, "PNG")


def load_rgba_png(path: str) -> np.ndarray:
    image = QImage(path).convertToFormat(QImage.Format.Format_RGBA8888)
    w, h = image.width(), image.height()
    ptr = image.bits()
    arr = np.frombuffer(ptr, dtype=np.uint8, count=h * w * 4).reshape((h, w, 4)).copy()
    return arr


# ---------------------------------------------------------------------------
# Photoshop-style dual (foreground/background) color swatch
# ---------------------------------------------------------------------------

class ColorSwatchWidget(QWidget):
    """Overlapping foreground/background color squares with swap and reset controls,
    in the classic Photoshop layout.
    """

    def __init__(self, main_window: "MainWindow", parent=None):
        super().__init__(parent)
        self.main_window = main_window
        self.setFixedSize(40, 58)

        squares = QWidget(self)
        squares.setFixedSize(36, 36)

        self.fg_btn = QToolButton(squares)
        self.fg_btn.setGeometry(0, 0, 24, 24)
        self.fg_btn.clicked.connect(self._pick_foreground)

        self.bg_btn = QToolButton(squares)
        self.bg_btn.setGeometry(12, 12, 24, 24)
        self.bg_btn.clicked.connect(self._pick_background)
        self.bg_btn.lower()
        self.fg_btn.raise_()

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(4)
        outer.addWidget(squares)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(2)
        self.swap_btn = QToolButton()
        self.swap_btn.setIcon(icon("swap_colors"))
        self.swap_btn.setIconSize(QSize(12, 12))
        self.swap_btn.setFixedSize(17, 17)
        self.swap_btn.setToolTip("Swap foreground / background (X)")
        self.swap_btn.clicked.connect(self._swap)
        self.reset_btn = QToolButton()
        self.reset_btn.setIcon(icon("reset_colors"))
        self.reset_btn.setIconSize(QSize(12, 12))
        self.reset_btn.setFixedSize(17, 17)
        self.reset_btn.setToolTip("Reset to black / white (D)")
        self.reset_btn.clicked.connect(self._reset)
        btn_row.addWidget(self.swap_btn)
        btn_row.addWidget(self.reset_btn)
        outer.addLayout(btn_row)

        self.refresh()

    def _swatch_style(self, checked_border: bool) -> str:
        border = "2px solid #d8d8d8" if checked_border else "1px solid #1c1c1e"
        return f"QToolButton {{ border: {border}; background: transparent; }}"

    def refresh(self) -> None:
        fg = self.main_window.canvas.primary_color
        bg = self.main_window.canvas.secondary_color
        self.fg_btn.setStyleSheet(
            f"QToolButton {{ background-color: rgba({fg[0]},{fg[1]},{fg[2]},{fg[3]}); border: 1px solid #1c1c1e; }}"
        )
        self.bg_btn.setStyleSheet(
            f"QToolButton {{ background-color: rgba({bg[0]},{bg[1]},{bg[2]},{bg[3]}); border: 1px solid #1c1c1e; }}"
        )

    def _pick_foreground(self) -> None:
        c = self.main_window.canvas.primary_color
        color = QColorDialog.getColor(QColor(*c), self, "Foreground Color", QColorDialog.ColorDialogOption.ShowAlphaChannel)
        if color.isValid():
            self.main_window.set_primary_color((color.red(), color.green(), color.blue(), color.alpha()))

    def _pick_background(self) -> None:
        c = self.main_window.canvas.secondary_color
        color = QColorDialog.getColor(QColor(*c), self, "Background Color", QColorDialog.ColorDialogOption.ShowAlphaChannel)
        if color.isValid():
            self.main_window.set_secondary_color((color.red(), color.green(), color.blue(), color.alpha()))

    def _swap(self) -> None:
        canvas = self.main_window.canvas
        canvas.primary_color, canvas.secondary_color = canvas.secondary_color, canvas.primary_color
        self.refresh()

    def _reset(self) -> None:
        self.main_window.set_primary_color((0, 0, 0, 255))
        self.main_window.set_secondary_color((255, 255, 255, 255))


# ---------------------------------------------------------------------------
# Docks
# ---------------------------------------------------------------------------

class ToolDock(QDockWidget):
    """Narrow, Photoshop-style vertical icon strip: one tool per row, brush size
    stepper, a navigator toggle, and the dual color swatch pinned at the bottom.
    """

    # Note: ToolKind.BRUSH has no dedicated sidebar button — picking a brush in the
    # Custom Brush Manager panel activates it directly (see BrushManagerPanel._select_brush).
    TOOL_ICONS = {
        ToolKind.PENCIL: "pencil",
        ToolKind.ERASER: "eraser",
        ToolKind.BUCKET: "bucket",
        ToolKind.SELECT_RECT: "select_rect",
        ToolKind.SELECT_LASSO: "select_lasso",
        ToolKind.EYEDROPPER: "eyedropper",
        ToolKind.HAND: "hand",
        ToolKind.ZOOM: "zoom",
    }

    TOOL_LABELS = {
        ToolKind.PENCIL: "Pencil",
        ToolKind.BRUSH: "Custom Brush",
        ToolKind.ERASER: "Eraser",
        ToolKind.BUCKET: "Bucket",
        ToolKind.SELECT_RECT: "Select (Rect)",
        ToolKind.SELECT_LASSO: "Select (Lasso)",
        ToolKind.EYEDROPPER: "Eyedropper",
        ToolKind.HAND: "Hand (Pan)",
        ToolKind.ZOOM: "Zoom (Right-click to zoom out)",
    }

    def __init__(self, main_window: "MainWindow"):
        super().__init__("Tools", main_window)
        self.main_window = main_window
        self.setTitleBarWidget(QWidget())  # collapse the title bar for a slim strip
        self.setFeatures(QDockWidget.DockWidgetFeature.NoDockWidgetFeatures)

        container = QWidget()
        container.setFixedWidth(44)
        layout = QVBoxLayout(container)
        layout.setContentsMargins(4, 6, 4, 6)
        layout.setSpacing(3)

        self.group = QButtonGroup(self)
        self.group.setExclusive(True)

        for kind in self.TOOL_ICONS:
            btn = QToolButton()
            btn.setCheckable(True)
            btn.setIcon(icon(self.TOOL_ICONS[kind]))
            btn.setIconSize(QSize(18, 18))
            btn.setToolTip(self.TOOL_LABELS[kind])
            btn.setFixedSize(34, 30)
            btn.clicked.connect(lambda checked, k=kind: self._select_tool(k))
            self.group.addButton(btn)
            layout.addWidget(btn)
        self.group.buttons()[0].setChecked(True)

        layout.addStretch(1)

        self.navigator_btn = QToolButton()
        self.navigator_btn.setCheckable(True)
        self.navigator_btn.setChecked(True)
        self.navigator_btn.setIcon(icon("eye"))
        self.navigator_btn.setIconSize(QSize(16, 16))
        self.navigator_btn.setFixedSize(34, 28)
        self.navigator_btn.setToolTip("Toggle navigator preview")
        self.navigator_btn.toggled.connect(main_window.canvas.set_navigator_visible)
        layout.addWidget(self.navigator_btn)

        self.color_swatch = ColorSwatchWidget(main_window)
        layout.addWidget(self.color_swatch)

        container.setLayout(layout)
        self.setWidget(container)

    def _select_tool(self, kind: ToolKind) -> None:
        self.main_window.set_tool(kind)

    def set_color_swatch(self, rgba: tuple[int, int, int, int]) -> None:
        self.color_swatch.refresh()


class ToolOptionsBar(QWidget):
    """Contextual, horizontal tool-options strip for the top toolbar: shows only
    the controls relevant to whichever tool is currently active (Photoshop-style
    Options Bar), instead of one fixed set for every tool.
    """

    def __init__(self, main_window: "MainWindow"):
        super().__init__()
        self.main_window = main_window
        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 0, 6, 0)
        layout.setSpacing(10)

        self.pixel_perfect_chk = QCheckBox("Pixel-Perfect Line")
        self.pixel_perfect_chk.setChecked(True)
        self.pixel_perfect_chk.stateChanged.connect(
            lambda s: setattr(main_window.canvas, "pixel_perfect", bool(s))
        )
        layout.addWidget(self.pixel_perfect_chk)

        self.opacity_label = QLabel("Opacity:")
        layout.addWidget(self.opacity_label)
        self.brush_opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self.brush_opacity_slider.setRange(0, 100)
        self.brush_opacity_slider.setValue(100)
        self.brush_opacity_slider.setFixedWidth(100)
        self.brush_opacity_slider.valueChanged.connect(
            lambda v: main_window.canvas.set_brush_opacity(v / 100.0)
        )
        layout.addWidget(self.brush_opacity_slider)

        self.tolerance_label = QLabel("Tolerance:")
        layout.addWidget(self.tolerance_label)
        self.tolerance_slider = QSlider(Qt.Orientation.Horizontal)
        self.tolerance_slider.setRange(0, 255)
        self.tolerance_slider.setFixedWidth(100)
        self.tolerance_slider.valueChanged.connect(
            lambda v: setattr(main_window.canvas, "fill_tolerance", v)
        )
        layout.addWidget(self.tolerance_slider)

        self.sample_all_chk = QCheckBox("Sample All Layers")
        self.sample_all_chk.stateChanged.connect(
            lambda s: setattr(main_window.canvas, "sample_all_layers", bool(s))
        )
        layout.addWidget(self.sample_all_chk)

        self.no_options_label = QLabel("No options for this tool")
        self.no_options_label.setStyleSheet("color: #777;")
        layout.addWidget(self.no_options_label)

        layout.addStretch(1)
        self.update_for_tool(ToolKind.PENCIL)

    TOOL_OPTION_WIDGETS = {
        ToolKind.PENCIL: ("pixel_perfect_chk", "opacity_label", "brush_opacity_slider"),
        ToolKind.BRUSH: ("opacity_label", "brush_opacity_slider"),
        ToolKind.ERASER: ("pixel_perfect_chk",),
        ToolKind.BUCKET: ("tolerance_label", "tolerance_slider", "sample_all_chk"),
    }

    def update_for_tool(self, kind: ToolKind) -> None:
        relevant = self.TOOL_OPTION_WIDGETS.get(kind, ())
        all_widgets = (
            "pixel_perfect_chk", "opacity_label", "brush_opacity_slider",
            "tolerance_label", "tolerance_slider", "sample_all_chk",
        )
        for name in all_widgets:
            getattr(self, name).setVisible(name in relevant)
        self.no_options_label.setVisible(len(relevant) == 0)


class PaletteSwatchButton(QToolButton):
    """A plain color box: left click sets the foreground color, right click sets
    the background color, matching Photoshop-style swatch grid behavior.
    """

    left_clicked = Signal()
    right_clicked = Signal()

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.RightButton:
            self.right_clicked.emit()
        else:
            super().mousePressEvent(event)
            self.left_clicked.emit()


class PalettePanel(QDockWidget):
    SWATCH_SIZE = 22
    COLUMNS = 8

    def __init__(self, main_window: "MainWindow"):
        super().__init__("Palette", main_window)
        self.main_window = main_window
        self.palette = Palette.default()
        self.selected_index: int = -1

        container = QWidget()
        layout = QVBoxLayout(container)

        self.grid_widget = QWidget()
        self.grid = QGridLayout(self.grid_widget)
        self.grid.setSpacing(2)
        self.grid.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.grid_widget)
        layout.addStretch(1)

        btn_row = QHBoxLayout()
        add_btn = QPushButton("+")
        add_btn.setFixedSize(28, 24)
        add_btn.setToolTip("Add current foreground color to palette")
        add_btn.clicked.connect(self._add_color)
        remove_btn = QPushButton("−")
        remove_btn.setFixedSize(28, 24)
        remove_btn.setToolTip("Remove selected color")
        remove_btn.clicked.connect(self._remove_color)
        btn_row.addWidget(add_btn)
        btn_row.addWidget(remove_btn)
        btn_row.addStretch(1)
        layout.addLayout(btn_row)

        io_row = QHBoxLayout()
        import_btn = QPushButton("Import")
        import_btn.clicked.connect(self._import)
        export_btn = QPushButton("Export")
        export_btn.clicked.connect(self._export)
        io_row.addWidget(import_btn)
        io_row.addWidget(export_btn)
        layout.addLayout(io_row)

        container.setLayout(layout)
        self.setWidget(container)
        self.refresh()

    def refresh(self) -> None:
        while self.grid.count():
            item = self.grid.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()

        for i, rgba in enumerate(self.palette.colors):
            btn = PaletteSwatchButton()
            btn.setFixedSize(self.SWATCH_SIZE, self.SWATCH_SIZE)
            selected = i == self.selected_index
            border = "2px solid #d8d8d8" if selected else "1px solid #1c1c1e"
            btn.setStyleSheet(
                f"QToolButton {{ background-color: rgba({rgba[0]},{rgba[1]},{rgba[2]},{rgba[3]}); "
                f"border: {border}; border-radius: 2px; }}"
            )
            btn.left_clicked.connect(lambda idx=i: self._select_color(idx, primary=True))
            btn.right_clicked.connect(lambda idx=i: self._select_color(idx, primary=False))
            self.grid.addWidget(btn, i // self.COLUMNS, i % self.COLUMNS)

    def _select_color(self, idx: int, primary: bool) -> None:
        self.selected_index = idx
        rgba = self.palette.colors[idx]
        if primary:
            self.main_window.set_primary_color(rgba)
        else:
            self.main_window.set_secondary_color(rgba)
        self.refresh()

    def _add_color(self) -> None:
        rgba = self.main_window.canvas.primary_color
        before = len(self.palette.colors)
        idx = self.palette.add(rgba)
        if len(self.palette.colors) == before:
            self.main_window.statusBar().showMessage("Color already in palette", 2000)
        self.selected_index = idx
        self.refresh()

    def _remove_color(self) -> None:
        if self.selected_index >= 0:
            self.palette.remove(self.selected_index)
            self.selected_index = -1
            self.refresh()

    def _import(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Import Palette", "", "Palettes (*.gpl *.hex)")
        if not path:
            return
        self.palette = Palette.load_gpl(path) if path.endswith(".gpl") else Palette.load_hex(path)
        self.refresh()

    def _export(self) -> None:
        path, filter_str = QFileDialog.getSaveFileName(self, "Export Palette", "", "GIMP Palette (*.gpl);;Hex List (*.hex)")
        if not path:
            return
        if path.endswith(".hex"):
            self.palette.save_hex(path)
        else:
            self.palette.save_gpl(path)


def _render_matrix_thumbnail(matrix: np.ndarray, size: int = 26, on_dark: bool = True) -> QPixmap:
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


def _render_brush_thumbnail(brush: Brush, size: int = 26) -> QPixmap:
    return _render_matrix_thumbnail(brush.matrix, size)


def _render_layer_thumbnail(pixels: np.ndarray, size: int = 28) -> QPixmap:
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


class BrushSwatchButton(QToolButton):
    """A brush thumbnail: left click selects (and activates the Brush tool),
    right click deletes it.
    """

    left_clicked = Signal()
    right_clicked = Signal()

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.RightButton:
            self.right_clicked.emit()
        else:
            super().mousePressEvent(event)
            self.left_clicked.emit()


class BrushManagerPanel(QDockWidget):
    SWATCH_SIZE = 34
    COLUMNS = 6

    def __init__(self, main_window: "MainWindow"):
        super().__init__("Custom Brush Manager", main_window)
        self.main_window = main_window
        container = QWidget()
        layout = QVBoxLayout(container)

        self.grid_widget = QWidget()
        self.grid = QGridLayout(self.grid_widget)
        self.grid.setSpacing(3)
        self.grid.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.grid_widget)

        edit_btn = QPushButton("New / Edit Brush Matrix...")
        edit_btn.clicked.connect(self._open_editor)
        layout.addWidget(edit_btn)

        dither_row = QHBoxLayout()
        for size in (2, 4, 8):
            btn = QPushButton(f"Dither {size}x{size}")
            btn.clicked.connect(lambda _, s=size: self._add_dither(s))
            dither_row.addWidget(btn)
        layout.addLayout(dither_row)

        io_row = QHBoxLayout()
        load_btn = QPushButton("Load .pxbrush")
        load_btn.clicked.connect(self._load_brush)
        io_row.addWidget(load_btn)
        layout.addLayout(io_row)

        container.setLayout(layout)
        self.setWidget(container)
        self.refresh()

    def refresh(self) -> None:
        while self.grid.count():
            item = self.grid.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()

        manager = self.main_window.canvas.brush_manager
        for i, brush in enumerate(manager.brushes):
            btn = BrushSwatchButton()
            btn.setFixedSize(self.SWATCH_SIZE, self.SWATCH_SIZE)
            btn.setIcon(QIcon(_render_brush_thumbnail(brush)))
            btn.setIconSize(QSize(self.SWATCH_SIZE - 8, self.SWATCH_SIZE - 8))
            btn.setToolTip(f"{brush.name} ({brush.width}x{brush.height})\nRight-click to delete")
            selected = i == manager.active_index
            border = "2px solid #d8d8d8" if selected else "1px solid #1c1c1e"
            btn.setStyleSheet(
                f"QToolButton {{ background-color: #232326; border: {border}; border-radius: 2px; }}"
            )
            btn.left_clicked.connect(lambda idx=i: self._select_brush(idx))
            btn.right_clicked.connect(lambda idx=i: self._delete_brush(idx))
            self.grid.addWidget(btn, i // self.COLUMNS, i % self.COLUMNS)

    def _select_brush(self, idx: int) -> None:
        self.main_window.canvas.brush_manager.active_index = idx
        self.main_window.set_tool(ToolKind.BRUSH)
        self.refresh()

    def _delete_brush(self, idx: int) -> None:
        manager = self.main_window.canvas.brush_manager
        if len(manager.brushes) <= 1:
            self.main_window.statusBar().showMessage("At least one brush must remain", 2000)
            return
        manager.remove(idx)
        self.refresh()

    def _open_editor(self) -> None:
        dlg = BrushMatrixEditor(size=8, parent=self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            brush = dlg.result_brush()
            self.main_window.canvas.brush_manager.add(brush)
            self.refresh()

    def _add_dither(self, size: int) -> None:
        brush = make_dither_brush(size, 0.5)
        self.main_window.canvas.brush_manager.add(brush)
        self.refresh()

    def _load_brush(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Load Brush", "", "Pixel Brush (*.pxbrush)")
        if path:
            brush = Brush.load(path)
            self.main_window.canvas.brush_manager.add(brush)
            self.refresh()


class LayerStackPanel(QDockWidget):
    def __init__(self, main_window: "MainWindow"):
        super().__init__("Layers", main_window)
        self.main_window = main_window
        container = QWidget()
        layout = QVBoxLayout(container)

        self.list = QListWidget()
        self.list.setIconSize(QSize(28, 28))
        self.list.itemClicked.connect(self._select_layer)
        layout.addWidget(self.list)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(4)
        add_btn = QToolButton()
        add_btn.setIcon(icon("layer_add"))
        add_btn.setIconSize(ICON_SIZE)
        add_btn.setFixedSize(30, 30)
        add_btn.setToolTip("Add Layer")
        add_btn.clicked.connect(self._add_layer)

        remove_btn = QToolButton()
        remove_btn.setIcon(icon("layer_remove"))
        remove_btn.setIconSize(ICON_SIZE)
        remove_btn.setFixedSize(30, 30)
        remove_btn.setToolTip("Remove Layer")
        remove_btn.clicked.connect(self._remove_layer)

        self.vis_btn = QToolButton()
        self.vis_btn.setIcon(icon("eye"))
        self.vis_btn.setIconSize(ICON_SIZE)
        self.vis_btn.setFixedSize(30, 30)
        self.vis_btn.setToolTip("Toggle Visibility")
        self.vis_btn.clicked.connect(self._toggle_visible)

        up_btn = QToolButton()
        up_btn.setIcon(icon("arrow_up"))
        up_btn.setIconSize(ICON_SIZE)
        up_btn.setFixedSize(30, 30)
        up_btn.setToolTip("Move Layer Up")
        up_btn.clicked.connect(lambda: self._move_layer(1))

        down_btn = QToolButton()
        down_btn.setIcon(icon("arrow_down"))
        down_btn.setIconSize(ICON_SIZE)
        down_btn.setFixedSize(30, 30)
        down_btn.setToolTip("Move Layer Down")
        down_btn.clicked.connect(lambda: self._move_layer(-1))

        btn_row.addWidget(add_btn)
        btn_row.addWidget(remove_btn)
        btn_row.addWidget(self.vis_btn)
        btn_row.addWidget(up_btn)
        btn_row.addWidget(down_btn)
        btn_row.addStretch(1)
        layout.addLayout(btn_row)

        layout.addWidget(QLabel("Opacity:"))
        self.opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self.opacity_slider.setRange(0, 100)
        self.opacity_slider.setValue(100)
        self.opacity_slider.valueChanged.connect(self._set_opacity)
        layout.addWidget(self.opacity_slider)

        layout.addWidget(QLabel("Blend Mode:"))
        self.blend_combo = QComboBox()
        self.blend_combo.addItems([m.value for m in BlendMode])
        self.blend_combo.currentTextChanged.connect(self._set_blend_mode)
        layout.addWidget(self.blend_combo)

        container.setLayout(layout)
        self.setWidget(container)
        self.refresh()

    def refresh(self) -> None:
        self.list.clear()
        stack = self.main_window.layer_stack
        # Display the top-most rendered layer (highest index) first, matching the
        # conventional layer-panel order, even though the stack stores bottom-first.
        for idx in reversed(range(len(stack.layers))):
            layer = stack.layers[idx]
            vis = "◉" if layer.visible else "○"
            item = QListWidgetItem(f"{vis} {layer.name}  [{int(layer.opacity * 100)}%]")
            item.setIcon(QIcon(_render_layer_thumbnail(layer.pixels)))
            item.setData(Qt.ItemDataRole.UserRole, idx)
            self.list.addItem(item)
        active_row = (len(stack.layers) - 1) - stack.active_index
        if 0 <= active_row < self.list.count():
            self.list.setCurrentRow(active_row)
        self._update_vis_icon()
        self.main_window.canvas.mark_dirty()

    def _select_layer(self, item) -> None:
        idx = item.data(Qt.ItemDataRole.UserRole)
        self.main_window.layer_stack.active_index = idx
        layer = self.main_window.layer_stack.active
        self.opacity_slider.blockSignals(True)
        self.opacity_slider.setValue(int(layer.opacity * 100))
        self.opacity_slider.blockSignals(False)
        self.blend_combo.blockSignals(True)
        self.blend_combo.setCurrentText(layer.blend_mode.value)
        self.blend_combo.blockSignals(False)
        self._update_vis_icon()

    def _add_layer(self) -> None:
        self.main_window.layer_stack.add_layer()
        self.refresh()

    def _remove_layer(self) -> None:
        self.main_window.layer_stack.remove_layer(self.main_window.layer_stack.active_index)
        self.refresh()

    def _move_layer(self, delta: int) -> None:
        stack = self.main_window.layer_stack
        idx = stack.active_index
        new_idx = idx + delta
        if 0 <= new_idx < len(stack.layers):
            stack.move_layer(idx, new_idx)
            self.refresh()

    def _toggle_visible(self) -> None:
        layer = self.main_window.layer_stack.active
        layer.visible = not layer.visible
        self.refresh()

    def _update_vis_icon(self) -> None:
        layer = self.main_window.layer_stack.active
        self.vis_btn.setIcon(icon("eye") if layer.visible else icon("eye_off"))

    def _set_opacity(self, value: int) -> None:
        self.main_window.layer_stack.active.opacity = value / 100.0
        self.main_window.canvas.mark_dirty()
        self.refresh()

    def _set_blend_mode(self, text: str) -> None:
        for mode in BlendMode:
            if mode.value == text:
                self.main_window.layer_stack.active.blend_mode = mode
                break
        self.main_window.canvas.mark_dirty()


# ---------------------------------------------------------------------------
# Main Window
# ---------------------------------------------------------------------------

class MainWindow(QMainWindow):
    def __init__(self, width: int = 16, height: int = 16):
        super().__init__()
        self.setWindowTitle("OpenPixel")
        self.resize(1400, 900)
        self.project_name = "New Project"
        self.project_path: str | None = None

        self.layer_stack = LayerStack(width, height)
        self.canvas = PixelCanvas(self.layer_stack)
        self.canvas.set_stroke_commit_callback(self._on_stroke_committed)
        self.canvas.selection_changed.connect(self._on_selection_changed)
        self.canvas.color_picked.connect(self._on_color_picked)
        self.canvas.zoom_changed.connect(self._on_zoom_changed)
        self.canvas.transform_requested.connect(self._open_transform_dialog)
        self.setCentralWidget(self.canvas)

        self.undo_stack = QUndoStack(self)

        self.tool_dock = ToolDock(self)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.tool_dock)

        self.tool_options_dock = ToolOptionsDock(self)
        self.palette_panel = PalettePanel(self)
        self.brush_panel = BrushManagerPanel(self)
        self.layer_panel = LayerStackPanel(self)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.tool_options_dock)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.palette_panel)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.brush_panel)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.layer_panel)
        self.canvas.brush_created.connect(self.brush_panel.refresh)

        self._build_menus()
        self._build_toolbar()
        self._build_status_bar()

    # ---- Menus ----

    def _build_menus(self) -> None:
        menubar = self.menuBar()

        file_menu = menubar.addMenu("&File")
        file_menu.addAction(self._act("New Canvas...", self._new_canvas))
        file_menu.addAction(self._act("Open...", self._open_file))
        file_menu.addAction(self._act("Import PNG...", self._import_png))
        file_menu.addSeparator()
        file_menu.addAction(self._act("Save Project...", self._save_project))
        file_menu.addAction(self._act("Open Project...", self._open_project))
        file_menu.addSeparator()
        file_menu.addAction(self._act("Export PNG...", self._export_png))
        file_menu.addAction(self._act("Sprite Sheet Exporter...", self._open_sprite_exporter))
        file_menu.addSeparator()
        file_menu.addAction(self._act("Exit", self.close))

        edit_menu = menubar.addMenu("&Edit")
        self.undo_action = self.undo_stack.createUndoAction(self, "Undo")
        self.undo_action.setShortcut("Ctrl+Z")
        self.undo_action.setIcon(icon("undo"))
        self.redo_action = self.undo_stack.createRedoAction(self, "Redo")
        self.redo_action.setShortcut("Ctrl+Y")
        self.redo_action.setIcon(icon("redo"))
        edit_menu.addAction(self.undo_action)
        edit_menu.addAction(self.redo_action)
        edit_menu.addSeparator()
        edit_menu.addAction(self._act("Deselect", self._deselect))
        edit_menu.addAction(self._act("Transform Selection...", self._open_transform_dialog))

        view_menu = menubar.addMenu("&View")
        tiling_action = QAction("Tiling Mode Preview", self, checkable=True)
        tiling_action.toggled.connect(self._toggle_tiling)
        view_menu.addAction(tiling_action)
        grid_action = QAction("Show Pixel Grid", self, checkable=True)
        grid_action.setChecked(True)
        grid_action.toggled.connect(self.canvas.set_grid_enabled)
        view_menu.addAction(grid_action)
        view_menu.addAction(self._act("Fit to Window", self.canvas.fit_to_window))

        panels_menu = view_menu.addMenu("Panels")
        for dock in (self.tool_options_dock, self.palette_panel, self.brush_panel, self.layer_panel):
            action = dock.toggleViewAction()
            action.setText(dock.windowTitle())
            panels_menu.addAction(action)

        filters_menu = menubar.addMenu("Fi&lters")
        filters_menu.addAction(self._act("Palette Remap (Quantize)...", self._filter_quantize))
        filters_menu.addAction(self._act("Outline Generator...", self._filter_outline))
        filters_menu.addAction(self._act("Color Ramp Shading...", self._filter_ramp))
        filters_menu.addAction(self._act("Auto-Clean Double Pixels", self._filter_autoclean))
        filters_menu.addAction(self._act("Drop Shadow...", self._filter_dropshadow))

    def _act(self, label: str, slot) -> QAction:
        action = QAction(label, self)
        action.triggered.connect(slot)
        return action

    SIZE_RELEVANT_TOOLS = (ToolKind.PENCIL, ToolKind.ERASER)

    def _build_toolbar(self) -> None:
        toolbar = QToolBar("Main Toolbar", self)
        toolbar.setIconSize(ICON_SIZE)
        toolbar.setMovable(False)

        self.brush_size_widget = QWidget()
        size_row = QHBoxLayout(self.brush_size_widget)
        size_row.setContentsMargins(4, 0, 8, 0)
        size_row.setSpacing(2)
        size_down_btn = QToolButton()
        size_down_btn.setIcon(icon("minus"))
        size_down_btn.setIconSize(QSize(12, 12))
        size_down_btn.setFixedSize(22, 22)
        size_down_btn.setToolTip("Decrease brush size")
        size_down_btn.clicked.connect(lambda: self._step_brush_size(-1))
        self.brush_size_label = QLabel("1px")
        self.brush_size_label.setFixedWidth(28)
        self.brush_size_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        size_up_btn = QToolButton()
        size_up_btn.setIcon(icon("plus"))
        size_up_btn.setIconSize(QSize(12, 12))
        size_up_btn.setFixedSize(22, 22)
        size_up_btn.setToolTip("Increase brush size")
        size_up_btn.clicked.connect(lambda: self._step_brush_size(1))
        size_row.addWidget(size_down_btn)
        size_row.addWidget(self.brush_size_label)
        size_row.addWidget(size_up_btn)
        toolbar.addWidget(self.brush_size_widget)
        self._update_brush_size_widget()

        toolbar.addSeparator()
        toolbar.addAction(self.undo_action)
        toolbar.addAction(self.redo_action)
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, toolbar)

    def _step_brush_size(self, delta: int) -> None:
        self.canvas.set_pencil_size(self.canvas.pencil_size + delta)
        self._update_brush_size_widget()

    def _update_brush_size_widget(self) -> None:
        self.brush_size_widget.setVisible(self.canvas.active_tool_kind in self.SIZE_RELEVANT_TOOLS)
        self.brush_size_label.setText(f"{self.canvas.pencil_size}px")

    def _build_status_bar(self) -> None:
        self.zoom_label = QLabel("Zoom: 800%")
        self.statusBar().addPermanentWidget(self.zoom_label)

    # ---- Tool / canvas wiring ----

    def set_tool(self, kind: ToolKind) -> None:
        self.canvas.set_tool(kind)
        self._update_brush_size_widget()

    def _on_stroke_committed(self, layer_index: int, before: np.ndarray, after: np.ndarray) -> None:
        if np.array_equal(before, after):
            return
        cmd = PaintCommand(self.layer_stack, layer_index, before, after, self.canvas, text="Stroke")
        self.undo_stack.push(cmd)
        self.layer_panel.refresh()

    def _on_selection_changed(self, selection: Selection) -> None:
        self.statusBar().showMessage("Selection updated", 2000)

    def _on_color_picked(self, rgba: tuple[int, int, int, int]) -> None:
        self.set_primary_color(rgba)

    def set_primary_color(self, rgba: tuple[int, int, int, int]) -> None:
        self.canvas.primary_color = rgba
        self.tool_dock.set_color_swatch(rgba)

    def set_secondary_color(self, rgba: tuple[int, int, int, int]) -> None:
        self.canvas.secondary_color = rgba
        self.tool_dock.set_color_swatch(rgba)

    def _on_zoom_changed(self, zoom: float) -> None:
        self.zoom_label.setText(f"Zoom: {int(zoom * 100)}%")

    def _toggle_tiling(self, checked: bool) -> None:
        self.canvas.tiling_preview = checked
        self.canvas.update()

    def _deselect(self) -> None:
        self.canvas.clear_selection()

    # ---- File actions ----

    def _new_canvas(self) -> None:
        dlg = NewDocumentDialog(self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        w, h, background, name = dlg.result_values()
        self.layer_stack = LayerStack(w, h)
        self.layer_stack.layers[0].pixels[:, :] = background
        self.canvas.layers = self.layer_stack
        self.canvas.selection = None
        self.project_name = name
        self.project_path = None
        self.canvas.fit_to_window()
        self.canvas.mark_dirty()
        self.layer_panel.refresh()
        self.undo_stack.clear()
        self.setWindowTitle(f"OpenPixel – {name}")

    def _open_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open", "", "All Supported (*.png *.opxproj);;PNG Images (*.png);;OpenPixel Project (*.opxproj)"
        )
        if not path:
            return
        if path.lower().endswith(".opxproj"):
            self._load_project_from(path)
        else:
            self._import_png_from(path)

    def _import_png(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Import PNG", "", "PNG Images (*.png)")
        if path:
            self._import_png_from(path)

    def _import_png_from(self, path: str) -> None:
        arr = load_rgba_png(path)
        h, w = arr.shape[:2]
        self.layer_stack = LayerStack(w, h)
        self.layer_stack.layers[0].pixels = arr
        self.canvas.layers = self.layer_stack
        self.project_name = os.path.splitext(os.path.basename(path))[0]
        self.project_path = None
        self.canvas.fit_to_window()
        self.canvas.mark_dirty()
        self.layer_panel.refresh()
        self.undo_stack.clear()
        self.setWindowTitle(f"OpenPixel – {self.project_name}")

    def _default_out_dir(self) -> str:
        out_dir = os.path.join(os.getcwd(), "out")
        os.makedirs(out_dir, exist_ok=True)
        return out_dir

    def _export_png(self) -> None:
        default_path = os.path.join(self._default_out_dir(), f"{self.project_name}.png")
        path, _ = QFileDialog.getSaveFileName(self, "Export PNG", default_path, "PNG Images (*.png)")
        if path:
            save_rgba_png(self.layer_stack.composite(), path)

    def _save_project(self) -> None:
        default_path = self.project_path or os.path.join(self._default_out_dir(), f"{self.project_name}.opxproj")
        path, _ = QFileDialog.getSaveFileName(self, "Save Project", default_path, "OpenPixel Project (*.opxproj)")
        if not path:
            return
        if not path.lower().endswith(".opxproj"):
            path += ".opxproj"
        save_project(path, self.layer_stack, self.palette_panel.palette, self.project_name)
        self.project_path = path
        self.statusBar().showMessage(f"Saved to {path}", 3000)

    def _open_project(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Open Project", "", "OpenPixel Project (*.opxproj)")
        if path:
            self._load_project_from(path)

    def _load_project_from(self, path: str) -> None:
        stack, palette, name = load_project(path)
        self.layer_stack = stack
        self.canvas.layers = self.layer_stack
        self.canvas.selection = None
        self.palette_panel.palette = palette
        self.palette_panel.selected_index = -1
        self.palette_panel.refresh()
        self.project_name = name
        self.project_path = path
        self.canvas.fit_to_window()
        self.canvas.mark_dirty()
        self.layer_panel.refresh()
        self.undo_stack.clear()
        self.setWindowTitle(f"OpenPixel – {name}")

    def _open_sprite_exporter(self) -> None:
        dlg = SpriteSheetExportDialog(self.layer_stack, self)
        dlg.exec()

    # ---- Selection transform ----

    def _open_transform_dialog(self) -> None:
        if self.canvas.selection is None or self.canvas.selection.is_empty():
            QMessageBox.warning(self, "No Selection", "Make a selection first.")
            return
        dlg = QDialog(self)
        dlg.setWindowTitle("Transform Selection")
        layout = QGridLayout(dlg)
        layout.addWidget(QLabel("Scale X:"), 0, 0)
        sx = QDoubleSpinBox(); sx.setRange(0.1, 10.0); sx.setValue(1.0)
        layout.addWidget(sx, 0, 1)
        layout.addWidget(QLabel("Scale Y:"), 1, 0)
        sy = QDoubleSpinBox(); sy.setRange(0.1, 10.0); sy.setValue(1.0)
        layout.addWidget(sy, 1, 1)
        layout.addWidget(QLabel("Rotate (deg):"), 2, 0)
        rot = QDoubleSpinBox(); rot.setRange(-360, 360); rot.setValue(0.0)
        layout.addWidget(rot, 2, 1)
        apply_btn = QPushButton("Apply")
        layout.addWidget(apply_btn, 3, 0, 1, 2)

        def apply():
            layer = self.layer_stack.active
            before = layer.pixels.copy()
            new_pixels, new_mask, (ox, oy) = transform_selection(
                layer.pixels, self.canvas.selection.mask, sx.value(), sy.value(), rot.value()
            )
            layer.pixels[self.canvas.selection.mask] = 0
            h, w = new_pixels.shape[:2]
            lh, lw = layer.pixels.shape[:2]
            x0, y0 = max(0, ox), max(0, oy)
            x1, y1 = min(lw, ox + w), min(lh, oy + h)
            if x1 > x0 and y1 > y0:
                src = new_pixels[y0 - oy:y1 - oy, x0 - ox:x1 - ox]
                dst = layer.pixels[y0:y1, x0:x1]
                mask = src[..., 3] > 0
                dst[mask] = src[mask]
            self.canvas.mark_dirty()
            self._on_stroke_committed(self.layer_stack.active_index, before, layer.pixels.copy())
            dlg.accept()

        apply_btn.clicked.connect(apply)
        dlg.exec()

    # ---- Filters ----

    def _current_layer_edit(self, transform_fn) -> None:
        layer = self.layer_stack.active
        before = layer.pixels.copy()
        layer.pixels = transform_fn(layer.pixels)
        self.canvas.mark_dirty()
        self._on_stroke_committed(self.layer_stack.active_index, before, layer.pixels.copy())

    def _filter_quantize(self) -> None:
        self._current_layer_edit(lambda px: flt.apply_palette_quantize(px, self.palette_panel.palette))

    def _filter_outline(self) -> None:
        def op(px):
            outline = flt.generate_outline(px, color=(0, 0, 0, 255))
            base = outline.copy()
            mask = px[..., 3] > 0
            base[mask] = px[mask]
            return base
        self._current_layer_edit(op)

    def _filter_ramp(self) -> None:
        ramp = self.palette_panel.palette.colors
        shift, ok = QInputDialog.getInt(self, "Color Ramp Shading", "Shift steps (+darker / -lighter):", 1, -16, 16)
        if ok:
            self._current_layer_edit(lambda px: flt.apply_color_ramp_shift(px, ramp, shift))

    def _filter_autoclean(self) -> None:
        mask = self.canvas.selection.mask if self.canvas.selection and not self.canvas.selection.is_empty() else None
        self._current_layer_edit(lambda px: flt.auto_clean_double_pixels(px, mask))

    def _filter_dropshadow(self) -> None:
        self._current_layer_edit(lambda px: flt.apply_drop_shadow(px))


def main() -> None:
    app = QApplication(sys.argv)
    app.setStyleSheet(DARK_STYLESHEET)
    window = MainWindow(width=16, height=16)
    window.show()
    window.canvas.fit_to_window()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
