"""High-performance PyOpenGL canvas: orthographic projection, nearest-neighbor
texture filtering, smooth pan/zoom, pixel grid overlay, and 3x3 tiling preview.
"""
from __future__ import annotations

import ctypes

import numpy as np
from OpenGL import GL
from PySide6.QtCore import QPoint, QPointF, QSize, Qt, Signal
from PySide6.QtGui import QMouseEvent, QWheelEvent, QKeyEvent, QImage, QPixmap
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtWidgets import QLabel, QWidget, QHBoxLayout, QSlider, QToolButton, QMenu

from layers import LayerStack
from tools import Selection, Tool, ToolContext, ToolKind, create_tool
from brushes import Brush, BrushManager
from icons import icon
from sprite_sheet import SpriteSheetInfo

VERTEX_SHADER = """
#version 330 core
layout(location = 0) in vec2 in_pos;
layout(location = 1) in vec2 in_uv;
uniform mat4 u_mvp;
out vec2 v_uv;
void main() {
    gl_Position = u_mvp * vec4(in_pos, 0.0, 1.0);
    v_uv = in_uv;
}
"""

FRAGMENT_SHADER = """
#version 330 core
in vec2 v_uv;
out vec4 frag_color;
uniform sampler2D u_tex;
void main() {
    frag_color = texture(u_tex, v_uv);
}
"""

GRID_VERTEX_SHADER = """
#version 330 core
layout(location = 0) in vec2 in_pos;
uniform mat4 u_mvp;
void main() { gl_Position = u_mvp * vec4(in_pos, 0.0, 1.0); }
"""

GRID_FRAGMENT_SHADER = """
#version 330 core
out vec4 frag_color;
uniform vec4 u_color;
void main() { frag_color = u_color; }
"""


def _ortho(left, right, bottom, top, near=-1.0, far=1.0) -> np.ndarray:
    m = np.identity(4, dtype=np.float32)
    m[0, 0] = 2.0 / (right - left)
    m[1, 1] = 2.0 / (top - bottom)
    m[2, 2] = -2.0 / (far - near)
    m[0, 3] = -(right + left) / (right - left)
    m[1, 3] = -(top + bottom) / (top - bottom)
    m[2, 3] = -(far + near) / (far - near)
    return m.T  # column-major for OpenGL


class PixelCanvas(QOpenGLWidget):
    """The main editing viewport."""

    color_picked = Signal(tuple)
    selection_changed = Signal(object)
    canvas_edited = Signal()
    zoom_changed = Signal(float)
    transform_requested = Signal()
    brush_created = Signal()
    undo_requested = Signal()
    redo_requested = Signal()

    GRID_VISIBLE_ZOOM_THRESHOLD = 8.0  # 800%
    TILE_PREVIEW_ALPHA = 0.35

    def __init__(self, layer_stack: LayerStack, parent=None):
        super().__init__(parent)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMouseTracking(True)

        self.layers = layer_stack
        self.brush_manager = BrushManager()

        self.zoom: float = 8.0
        self.pan = QPointF(0.0, 0.0)
        self.tiling_preview = False
        self.grid_enabled = True

        self.active_tool: Tool = create_tool(ToolKind.PENCIL)
        self._stroke_tool: Tool = self.active_tool
        self.active_tool_kind = ToolKind.PENCIL
        self.primary_color: tuple[int, int, int, int] = (0, 0, 0, 255)
        self.secondary_color: tuple[int, int, int, int] = (255, 255, 255, 255)
        self.pixel_perfect = True
        self.fill_tolerance = 0
        self.sample_all_layers = False
        self.selection: Selection | None = None
        self._lasso_points: list[tuple[int, int]] | None = None
        self._clipboard: dict | None = None
        self.sprite_sheet: SpriteSheetInfo | None = None

        self.pencil_size = 1
        self.pencil_brush = Brush.solid_square(1)
        self.brush_opacity = 1.0
        self._hover_pixel: tuple[int, int] | None = None

        self._panning = False
        self._space_held = False
        self._last_mouse_pos = QPoint()
        self._stroke_active = False
        self._undo_snapshot: np.ndarray | None = None
        self._on_stroke_committed = None  # callback(layer_index, before, after)

        self._program = None
        self._grid_program = None
        self._vao = None
        self._vbo = None
        self._texture_id = None
        self._grid_vao = None
        self._grid_vbo = None
        self._tex_dirty = True

        self.tool_label = QLabel(self)
        self.tool_label.setStyleSheet(
            "background-color: rgba(30,30,32,220); color: #e8e8e8; padding: 2px 6px;"
            "border-radius: 3px; font-size: 11px;"
        )
        self.tool_label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.tool_label.adjustSize()
        self.tool_label.hide()
        self._update_tool_label_text()

        self.NAVIGATOR_SIZE = 160
        self.navigator_visible = True
        self.navigator = QLabel(self)
        self.navigator.setFixedSize(self.NAVIGATOR_SIZE, self.NAVIGATOR_SIZE)
        self.navigator.setStyleSheet(
            "background-color: #ffffff; border: 1px solid #55555a; border-radius: 3px;"
        )
        self.navigator.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.navigator.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.navigator.setVisible(self.navigator_visible)

        self._build_zoom_bar()
        self._position_overlays()
        self.zoom_changed.connect(self._sync_zoom_controls)

    def _build_zoom_bar(self) -> None:
        self.zoom_bar = QWidget(self)
        self.zoom_bar.setStyleSheet(
            "QWidget { background-color: rgba(30,30,32,225); border: 1px solid #55555a; border-radius: 4px; }"
            "QToolButton { background: transparent; border: none; border-radius: 3px; }"
            "QToolButton:hover { background: rgba(255,255,255,30); }"
        )
        layout = QHBoxLayout(self.zoom_bar)
        layout.setContentsMargins(6, 4, 8, 4)
        layout.setSpacing(6)

        self.zoom_out_btn = QToolButton()
        self.zoom_out_btn.setIcon(icon("minus"))
        self.zoom_out_btn.setIconSize(QSize(14, 14))
        self.zoom_out_btn.setFixedSize(22, 22)
        self.zoom_out_btn.setToolTip("Zoom Out")
        self.zoom_out_btn.clicked.connect(lambda: self._step_zoom(1 / 1.25))
        layout.addWidget(self.zoom_out_btn)

        self.zoom_slider = QSlider(Qt.Orientation.Horizontal)
        self.zoom_slider.setRange(25, 3200)
        self.zoom_slider.setValue(int(self.zoom * 100))
        self.zoom_slider.setFixedWidth(110)
        self.zoom_slider.valueChanged.connect(self._on_zoom_slider)
        layout.addWidget(self.zoom_slider)

        self.zoom_in_btn = QToolButton()
        self.zoom_in_btn.setIcon(icon("plus"))
        self.zoom_in_btn.setIconSize(QSize(14, 14))
        self.zoom_in_btn.setFixedSize(22, 22)
        self.zoom_in_btn.setToolTip("Zoom In")
        self.zoom_in_btn.clicked.connect(lambda: self._step_zoom(1.25))
        layout.addWidget(self.zoom_in_btn)

        self.zoom_pct_label = QLabel(f"{int(self.zoom * 100)}%")
        self.zoom_pct_label.setFixedWidth(44)
        self.zoom_pct_label.setToolTip("Click to reset to 100%")
        self.zoom_pct_label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, False)
        self.zoom_pct_label.mousePressEvent = lambda e: self._step_zoom(1.0 / self.zoom)
        layout.addWidget(self.zoom_pct_label)

        separator = QLabel("|")
        separator.setStyleSheet("color: #55555a;")
        layout.addWidget(separator)

        self.undo_btn = QToolButton()
        self.undo_btn.setIcon(icon("undo"))
        self.undo_btn.setIconSize(QSize(14, 14))
        self.undo_btn.setFixedSize(22, 22)
        self.undo_btn.setToolTip("Undo")
        self.undo_btn.clicked.connect(self.undo_requested.emit)
        layout.addWidget(self.undo_btn)

        self.redo_btn = QToolButton()
        self.redo_btn.setIcon(icon("redo"))
        self.redo_btn.setIconSize(QSize(14, 14))
        self.redo_btn.setFixedSize(22, 22)
        self.redo_btn.setToolTip("Redo")
        self.redo_btn.clicked.connect(self.redo_requested.emit)
        layout.addWidget(self.redo_btn)

        self.zoom_bar.adjustSize()

    def _step_zoom(self, factor: float) -> None:
        center = QPointF(self.width() / 2.0, self.height() / 2.0)
        self._zoom_at(center, factor)

    def _on_zoom_slider(self, value: int) -> None:
        center = QPointF(self.width() / 2.0, self.height() / 2.0)
        factor = (value / 100.0) / self.zoom
        self._zoom_at(center, factor)

    def _sync_zoom_controls(self) -> None:
        pct = int(round(self.zoom * 100))
        self.zoom_slider.blockSignals(True)
        self.zoom_slider.setValue(pct)
        self.zoom_slider.blockSignals(False)
        self.zoom_pct_label.setText(f"{pct}%")

    def _position_overlays(self) -> None:
        margin = 10
        self.navigator.move(margin, self.height() - self.navigator.height() - margin)
        self.zoom_bar.move(
            self.width() - self.zoom_bar.width() - margin,
            self.height() - self.zoom_bar.height() - margin,
        )

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._position_overlays()

    def clear_selection(self) -> None:
        self.selection = None
        self._lasso_points = None
        self.update()

    def set_grid_enabled(self, enabled: bool) -> None:
        self.grid_enabled = enabled
        self.update()

    def set_sprite_sheet(self, sheet: SpriteSheetInfo | None) -> None:
        self.sprite_sheet = sheet
        self.update()

    def focus_on_frame(self, col: int, row: int) -> None:
        """Zoom/pan so a single sprite-sheet frame fills most of the viewport,
        for focused per-frame editing."""
        if self.sprite_sheet is None:
            return
        x0, y0, x1, y1 = self.sprite_sheet.frame_rect(col, row)
        fw, fh = x1 - x0, y1 - y0
        margin = 0.85
        zx = self.width() / max(fw, 1) * margin
        zy = self.height() / max(fh, 1) * margin
        self.zoom = float(np.clip(min(zx, zy), 0.25, 64.0))
        cw, ch = self.layers.width, self.layers.height
        frame_cx, frame_cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        self.pan = QPointF(frame_cx - cw / 2.0, frame_cy - ch / 2.0)
        self.zoom_changed.emit(self.zoom)
        self.update()

    def set_navigator_visible(self, visible: bool) -> None:
        self.navigator_visible = visible
        self.navigator.setVisible(visible)
        if visible:
            self._refresh_navigator()

    # ---- GL lifecycle ----

    def initializeGL(self) -> None:
        GL.glClearColor(0.16, 0.16, 0.18, 1.0)
        GL.glEnable(GL.GL_BLEND)
        GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)

        self._program = self._compile_program(VERTEX_SHADER, FRAGMENT_SHADER)
        self._grid_program = self._compile_program(GRID_VERTEX_SHADER, GRID_FRAGMENT_SHADER)

        self._vao = GL.glGenVertexArrays(1)
        self._vbo = GL.glGenBuffers(1)
        GL.glBindVertexArray(self._vao)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._vbo)
        GL.glEnableVertexAttribArray(0)
        GL.glVertexAttribPointer(0, 2, GL.GL_FLOAT, GL.GL_FALSE, 16, ctypes.c_void_p(0))
        GL.glEnableVertexAttribArray(1)
        GL.glVertexAttribPointer(1, 2, GL.GL_FLOAT, GL.GL_FALSE, 16, ctypes.c_void_p(8))
        GL.glBindVertexArray(0)

        self._texture_id = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self._texture_id)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_NEAREST)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_NEAREST)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)

        self._grid_vao = GL.glGenVertexArrays(1)
        self._grid_vbo = GL.glGenBuffers(1)

    def _compile_program(self, vs_src: str, fs_src: str):
        vs = GL.glCreateShader(GL.GL_VERTEX_SHADER)
        GL.glShaderSource(vs, vs_src)
        GL.glCompileShader(vs)
        if not GL.glGetShaderiv(vs, GL.GL_COMPILE_STATUS):
            raise RuntimeError(GL.glGetShaderInfoLog(vs).decode())

        fs = GL.glCreateShader(GL.GL_FRAGMENT_SHADER)
        GL.glShaderSource(fs, fs_src)
        GL.glCompileShader(fs)
        if not GL.glGetShaderiv(fs, GL.GL_COMPILE_STATUS):
            raise RuntimeError(GL.glGetShaderInfoLog(fs).decode())

        program = GL.glCreateProgram()
        GL.glAttachShader(program, vs)
        GL.glAttachShader(program, fs)
        GL.glLinkProgram(program)
        if not GL.glGetProgramiv(program, GL.GL_LINK_STATUS):
            raise RuntimeError(GL.glGetProgramInfoLog(program).decode())
        GL.glDeleteShader(vs)
        GL.glDeleteShader(fs)
        return program

    def resizeGL(self, w: int, h: int) -> None:
        GL.glViewport(0, 0, max(w, 1), max(h, 1))

    def mark_dirty(self) -> None:
        self._tex_dirty = True
        self.update()

    def _upload_texture(self) -> None:
        composite = self.layers.composite()
        h, w = composite.shape[:2]
        GL.glBindTexture(GL.GL_TEXTURE_2D, self._texture_id)
        GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 1)
        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA, w, h, 0, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, composite)
        self._tex_dirty = False
        if self.navigator_visible:
            self._refresh_navigator(composite)

    def _refresh_navigator(self, composite: np.ndarray | None = None) -> None:
        if composite is None:
            composite = self.layers.composite()
        composite = np.ascontiguousarray(composite)
        h, w = composite.shape[:2]
        image = QImage(composite.tobytes(), w, h, w * 4, QImage.Format.Format_RGBA8888)
        pixmap = QPixmap.fromImage(image).scaled(
            self.NAVIGATOR_SIZE - 8, self.NAVIGATOR_SIZE - 8,
            Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation,
        )
        self.navigator.setPixmap(pixmap)

    def paintGL(self) -> None:
        GL.glClear(GL.GL_COLOR_BUFFER_BIT)
        if self._tex_dirty:
            self._upload_texture()

        view_w, view_h = self.width(), self.height()
        cw, ch = self.layers.width, self.layers.height
        half_w = view_w / (2.0 * self.zoom)
        half_h = view_h / (2.0 * self.zoom)
        cx, cy = cw / 2.0 + self.pan.x(), ch / 2.0 + self.pan.y()
        mvp = _ortho(cx - half_w, cx + half_w, cy + half_h, cy - half_h)

        GL.glUseProgram(self._program)
        loc = GL.glGetUniformLocation(self._program, "u_mvp")
        GL.glUniformMatrix4fv(loc, 1, GL.GL_FALSE, mvp)
        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self._texture_id)
        GL.glUniform1i(GL.glGetUniformLocation(self._program, "u_tex"), 0)

        tiles = [(0, 0)]
        if self.tiling_preview:
            tiles = [(dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)]

        for i, (tx, ty) in enumerate(tiles):
            x0, y0 = tx * cw, ty * ch
            x1, y1 = x0 + cw, y0 + ch
            verts = np.array([
                x0, y0, 0.0, 0.0,
                x1, y0, 1.0, 0.0,
                x1, y1, 1.0, 1.0,
                x0, y0, 0.0, 0.0,
                x1, y1, 1.0, 1.0,
                x0, y1, 0.0, 1.0,
            ], dtype=np.float32)
            GL.glBindVertexArray(self._vao)
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._vbo)
            GL.glBufferData(GL.GL_ARRAY_BUFFER, verts.nbytes, verts, GL.GL_DYNAMIC_DRAW)
            is_center = (tx == 0 and ty == 0)
            if not is_center:
                GL.glBlendColor(1.0, 1.0, 1.0, self.TILE_PREVIEW_ALPHA)
            GL.glDrawArrays(GL.GL_TRIANGLES, 0, 6)
            GL.glBindVertexArray(0)

        if self.grid_enabled and self.zoom > self.GRID_VISIBLE_ZOOM_THRESHOLD:
            self._draw_pixel_grid(mvp, cw, ch)

        if self.sprite_sheet is not None:
            self._draw_sprite_sheet_grid(mvp)

        if self.selection is not None and not self.selection.is_empty():
            self._draw_selection_outline(mvp)

        if (self._hover_pixel is not None and not self._stroke_active
                and self.active_tool_kind in (ToolKind.PENCIL, ToolKind.BRUSH, ToolKind.ERASER)):
            self._draw_brush_preview(mvp)

    def _draw_brush_preview(self, mvp: np.ndarray) -> None:
        brush = self._current_paint_brush()
        cx, cy = self._hover_pixel
        bw, bh = brush.width, brush.height
        x0 = cx - bw // 2
        y0 = cy - bh // 2

        ys, xs = np.nonzero(brush.matrix)
        if len(xs) == 0:
            return
        fill_verts = []
        for mx, my in zip(xs.tolist(), ys.tolist()):
            px, py = x0 + mx, y0 + my
            fill_verts += [
                px, py, px + 1, py, px + 1, py + 1,
                px, py, px + 1, py + 1, px, py + 1,
            ]
        fill_arr = np.array(fill_verts, dtype=np.float32)

        GL.glUseProgram(self._grid_program)
        loc = GL.glGetUniformLocation(self._grid_program, "u_mvp")
        GL.glUniformMatrix4fv(loc, 1, GL.GL_FALSE, mvp)
        color_loc = GL.glGetUniformLocation(self._grid_program, "u_color")
        GL.glUniform4f(color_loc, 1.0, 1.0, 1.0, 0.35)

        GL.glBindVertexArray(self._grid_vao)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._grid_vbo)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, fill_arr.nbytes, fill_arr, GL.GL_DYNAMIC_DRAW)
        GL.glEnableVertexAttribArray(0)
        GL.glVertexAttribPointer(0, 2, GL.GL_FLOAT, GL.GL_FALSE, 8, ctypes.c_void_p(0))
        GL.glDrawArrays(GL.GL_TRIANGLES, 0, len(fill_verts) // 2)
        GL.glBindVertexArray(0)

        outline = np.array([
            x0, y0, x0 + bw, y0,
            x0 + bw, y0, x0 + bw, y0 + bh,
            x0 + bw, y0 + bh, x0, y0 + bh,
            x0, y0 + bh, x0, y0,
        ], dtype=np.float32)
        GL.glUniform4f(color_loc, 1.0, 1.0, 1.0, 0.9)
        GL.glBindVertexArray(self._grid_vao)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._grid_vbo)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, outline.nbytes, outline, GL.GL_DYNAMIC_DRAW)
        GL.glEnableVertexAttribArray(0)
        GL.glVertexAttribPointer(0, 2, GL.GL_FLOAT, GL.GL_FALSE, 8, ctypes.c_void_p(0))
        GL.glDrawArrays(GL.GL_LINES, 0, len(outline) // 2)
        GL.glBindVertexArray(0)

    def _draw_pixel_grid(self, mvp: np.ndarray, cw: int, ch: int) -> None:
        lines = []
        for x in range(cw + 1):
            lines += [x, 0, x, ch]
        for y in range(ch + 1):
            lines += [0, y, cw, y]
        verts = np.array(lines, dtype=np.float32)

        GL.glUseProgram(self._grid_program)
        loc = GL.glGetUniformLocation(self._grid_program, "u_mvp")
        GL.glUniformMatrix4fv(loc, 1, GL.GL_FALSE, mvp)
        color_loc = GL.glGetUniformLocation(self._grid_program, "u_color")
        GL.glUniform4f(color_loc, 1.0, 1.0, 1.0, 0.25)

        GL.glBindVertexArray(self._grid_vao)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._grid_vbo)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, verts.nbytes, verts, GL.GL_DYNAMIC_DRAW)
        GL.glEnableVertexAttribArray(0)
        GL.glVertexAttribPointer(0, 2, GL.GL_FLOAT, GL.GL_FALSE, 8, ctypes.c_void_p(0))
        GL.glDrawArrays(GL.GL_LINES, 0, len(lines) // 2)
        GL.glBindVertexArray(0)

    def _draw_sprite_sheet_grid(self, mvp: np.ndarray) -> None:
        """Thicker, high-contrast frame boundaries subdividing the canvas into
        sprite-sheet cells, always visible regardless of zoom level."""
        sheet = self.sprite_sheet
        lines = []
        for row in range(sheet.rows):
            for col in range(sheet.columns):
                x0, y0, x1, y1 = sheet.frame_rect(col, row)
                lines += [
                    x0, y0, x1, y0,
                    x1, y0, x1, y1,
                    x1, y1, x0, y1,
                    x0, y1, x0, y0,
                ]
        verts = np.array(lines, dtype=np.float32)

        GL.glUseProgram(self._grid_program)
        loc = GL.glGetUniformLocation(self._grid_program, "u_mvp")
        GL.glUniformMatrix4fv(loc, 1, GL.GL_FALSE, mvp)
        color_loc = GL.glGetUniformLocation(self._grid_program, "u_color")
        GL.glUniform4f(color_loc, 1.0, 0.55, 0.15, 0.9)

        GL.glLineWidth(2.0)
        GL.glBindVertexArray(self._grid_vao)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._grid_vbo)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, verts.nbytes, verts, GL.GL_DYNAMIC_DRAW)
        GL.glEnableVertexAttribArray(0)
        GL.glVertexAttribPointer(0, 2, GL.GL_FLOAT, GL.GL_FALSE, 8, ctypes.c_void_p(0))
        GL.glDrawArrays(GL.GL_LINES, 0, len(lines) // 2)
        GL.glBindVertexArray(0)
        GL.glLineWidth(1.0)

    def _draw_selection_outline(self, mvp: np.ndarray) -> None:
        if self._lasso_points is not None and len(self._lasso_points) >= 2:
            pts = self._lasso_points
            verts = []
            for i in range(len(pts)):
                x0, y0 = pts[i]
                x1, y1 = pts[(i + 1) % len(pts)]
                verts += [x0, y0, x1, y1]
            verts = np.array(verts, dtype=np.float32)
        else:
            bounds = self.selection.bounds()
            if bounds is None:
                return
            x0, y0, x1, y1 = bounds
            verts = np.array([
                x0, y0, x1, y0,
                x1, y0, x1, y1,
                x1, y1, x0, y1,
                x0, y1, x0, y0,
            ], dtype=np.float32)

        GL.glUseProgram(self._grid_program)
        loc = GL.glGetUniformLocation(self._grid_program, "u_mvp")
        GL.glUniformMatrix4fv(loc, 1, GL.GL_FALSE, mvp)
        color_loc = GL.glGetUniformLocation(self._grid_program, "u_color")
        GL.glUniform4f(color_loc, 1.0, 1.0, 0.2, 0.9)
        GL.glBindVertexArray(self._grid_vao)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._grid_vbo)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, verts.nbytes, verts, GL.GL_DYNAMIC_DRAW)
        GL.glEnableVertexAttribArray(0)
        GL.glVertexAttribPointer(0, 2, GL.GL_FLOAT, GL.GL_FALSE, 8, ctypes.c_void_p(0))
        GL.glDrawArrays(GL.GL_LINES, 0, len(verts) // 2)
        GL.glBindVertexArray(0)

    # ---- Coordinate mapping ----

    def widget_to_pixel(self, pos: QPointF) -> tuple[int, int]:
        view_w, view_h = self.width(), self.height()
        cw, ch = self.layers.width, self.layers.height
        half_w = view_w / (2.0 * self.zoom)
        half_h = view_h / (2.0 * self.zoom)
        cx, cy = cw / 2.0 + self.pan.x(), ch / 2.0 + self.pan.y()
        left, top = cx - half_w, cy - half_h
        world_x = left + pos.x() / self.zoom
        world_y = top + pos.y() / self.zoom
        return int(np.floor(world_x)), int(np.floor(world_y))

    # ---- Input handling ----

    def wheelEvent(self, event: QWheelEvent) -> None:
        factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        self._zoom_at(event.position(), factor)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.MiddleButton or (
            event.button() == Qt.MouseButton.LeftButton and self._space_held
        ) or (event.button() == Qt.MouseButton.LeftButton and self.active_tool_kind == ToolKind.HAND):
            self._panning = True
            self._last_mouse_pos = event.position().toPoint()
            return

        if self.active_tool_kind == ToolKind.ZOOM and event.button() in (
            Qt.MouseButton.LeftButton, Qt.MouseButton.RightButton
        ):
            factor = 1.5 if event.button() == Qt.MouseButton.LeftButton else 1 / 1.5
            self._zoom_at(event.position(), factor)
            return

        if event.button() == Qt.MouseButton.LeftButton:
            self._begin_stroke(event.position())

    def _zoom_at(self, widget_pos: QPointF, factor: float) -> None:
        before_x, before_y = self.widget_to_pixel(widget_pos)
        self.zoom = float(np.clip(self.zoom * factor, 0.25, 64.0))
        after_x, after_y = self.widget_to_pixel(widget_pos)
        self.pan += QPointF(before_x - after_x, before_y - after_y)
        self.zoom_changed.emit(self.zoom)
        self.update()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        self._position_tool_label(event.position())
        self._hover_pixel = self.widget_to_pixel(event.position())

        if self._panning:
            delta = event.position().toPoint() - self._last_mouse_pos
            self.pan -= QPointF(delta.x() / self.zoom, delta.y() / self.zoom)
            self._last_mouse_pos = event.position().toPoint()
            self.update()
            return

        if self._stroke_active:
            x, y = self.widget_to_pixel(event.position())
            self._tool_move(x, y)
        else:
            self.update()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() in (Qt.MouseButton.MiddleButton, Qt.MouseButton.LeftButton) and self._panning:
            self._panning = False
            return
        if self._stroke_active:
            x, y = self.widget_to_pixel(event.position())
            self._end_stroke(x, y)

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:
        if self.sprite_sheet is None or event.button() != Qt.MouseButton.LeftButton:
            return
        x, y = self.widget_to_pixel(event.position())
        frame = self.sprite_sheet.frame_at_pixel(x, y)
        if frame is not None:
            self.focus_on_frame(*frame)

    def enterEvent(self, event) -> None:
        self.tool_label.show()
        super().enterEvent(event)

    def leaveEvent(self, event) -> None:
        self.tool_label.hide()
        self._hover_pixel = None
        self.update()
        super().leaveEvent(event)

    def _position_tool_label(self, widget_pos: QPointF) -> None:
        x = int(widget_pos.x()) - self.tool_label.width() // 2
        y = int(widget_pos.y()) - self.tool_label.height() - 14
        self.tool_label.move(max(0, x), max(0, y))

    TOOL_NAMES = {
        ToolKind.PENCIL: "Pencil",
        ToolKind.BRUSH: "Custom Brush",
        ToolKind.ERASER: "Eraser",
        ToolKind.BUCKET: "Bucket",
        ToolKind.SELECT_RECT: "Select (Rect)",
        ToolKind.SELECT_LASSO: "Select (Lasso)",
        ToolKind.EYEDROPPER: "Eyedropper",
        ToolKind.HAND: "Hand",
        ToolKind.ZOOM: "Zoom",
    }

    def _update_tool_label_text(self) -> None:
        self.tool_label.setText(self.TOOL_NAMES.get(self.active_tool_kind, ""))
        self.tool_label.adjustSize()

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() == Qt.Key.Key_Space:
            self._space_held = True
        else:
            super().keyPressEvent(event)

    def keyReleaseEvent(self, event: QKeyEvent) -> None:
        if event.key() == Qt.Key.Key_Space:
            self._space_held = False
        else:
            super().keyReleaseEvent(event)

    # ---- Tool stroke lifecycle ----

    def _current_paint_brush(self) -> Brush:
        if self.active_tool_kind == ToolKind.BRUSH:
            return self.brush_manager.active
        return self.pencil_brush

    def set_pencil_size(self, size: int) -> None:
        self.pencil_size = max(1, min(64, size))
        self.pencil_brush = Brush.solid_square(self.pencil_size)
        self.update()

    def _paint_color(self) -> tuple[int, int, int, int]:
        r, g, b, a = self.primary_color
        if self.active_tool_kind in (ToolKind.PENCIL, ToolKind.BRUSH):
            a = int(round(a * self.brush_opacity))
        return (r, g, b, a)

    def set_brush_opacity(self, opacity: float) -> None:
        self.brush_opacity = max(0.0, min(1.0, opacity))

    def _make_context(self, color_override: tuple[int, int, int, int] | None = None) -> ToolContext:
        layer = self.layers.active
        return ToolContext(
            pixels=layer.pixels,
            color=color_override if color_override is not None else self._paint_color(),
            brush=self._current_paint_brush(),
            pixel_perfect=self.pixel_perfect,
            tolerance=self.fill_tolerance,
            sample_all=self.sample_all_layers,
            composite=self.layers.composite(),
            selection=self.selection,
        )

    def _begin_stroke(self, widget_pos: QPointF) -> None:
        x, y = self.widget_to_pixel(widget_pos)
        layer = self.layers.active
        if layer.locked:
            return
        self._undo_snapshot = layer.pixels.copy()
        self._stroke_active = True
        self._stroke_tool = self.active_tool
        self._ctx = self._make_context()
        self._stroke_tool.begin(self._ctx, x, y)
        self._after_tool_step()

    def _tool_move(self, x: int, y: int) -> None:
        self._stroke_tool.move(self._ctx, x, y)
        self._sync_selection_preview()
        self._after_tool_step()

    def _end_stroke(self, x: int, y: int) -> None:
        self._stroke_tool.end(self._ctx, x, y)
        self._sync_selection_preview()
        self._after_tool_step()
        self._stroke_active = False

        if self.active_tool_kind in (ToolKind.SELECT_RECT, ToolKind.SELECT_LASSO):
            self.selection = self._ctx.selection
            self._lasso_points = (
                list(self._ctx.stroke_points) if self.active_tool_kind == ToolKind.SELECT_LASSO else None
            )
            self.selection_changed.emit(self.selection)
            self.update()
        elif self.active_tool_kind == ToolKind.EYEDROPPER:
            if self._ctx.picked_color is not None:
                self.color_picked.emit(self._ctx.picked_color)
        else:
            if self._on_stroke_committed and self._undo_snapshot is not None:
                self._on_stroke_committed(self.layers.active_index, self._undo_snapshot, self.layers.active.pixels.copy())
        self._undo_snapshot = None

    def _sync_selection_preview(self) -> None:
        """Mirror the in-progress ToolContext selection onto the canvas so the
        rectangle/lasso marquee is visible live while the mouse is still down."""
        if self.active_tool_kind in (ToolKind.SELECT_RECT, ToolKind.SELECT_LASSO):
            self.selection = self._ctx.selection
            self._lasso_points = (
                list(self._ctx.stroke_points) if self.active_tool_kind == ToolKind.SELECT_LASSO else None
            )

    def _after_tool_step(self) -> None:
        if self._ctx.dirty_rects:
            self.mark_dirty()
        elif self.active_tool_kind in (ToolKind.SELECT_RECT, ToolKind.SELECT_LASSO):
            self.update()
        self.canvas_edited.emit()

    # ---- Selection context menu: copy / cut / paste / flip / rotate ----

    def contextMenuEvent(self, event) -> None:
        has_selection = self.selection is not None and not self.selection.is_empty()
        has_clipboard = self._clipboard is not None

        # Only ever show actions that actually do something right now, grouped
        # with separators — instead of listing everything and graying out
        # whatever doesn't apply to the current selection/clipboard state.
        groups: list[list[tuple[str, "callable"]]] = []

        edit_group = []
        if has_selection:
            edit_group.append(("Copy", self._copy_selection))
            edit_group.append(("Cut", self._cut_selection))
        if has_clipboard:
            edit_group.append(("Paste", self._paste_clipboard))
        if edit_group:
            groups.append(edit_group)

        transform_group = []
        if has_selection:
            transform_group.append(("Flip Horizontal", lambda: self._flip_selection(axis=1)))
            transform_group.append(("Flip Vertical", lambda: self._flip_selection(axis=0)))
            transform_group.append(("Rotate 90° CW", self._rotate_selection_90))
            transform_group.append(("Transform...", self.transform_requested.emit))
        if transform_group:
            groups.append(transform_group)

        selection_group = []
        if has_selection:
            selection_group.append(("Deselect", self.clear_selection))
            selection_group.append(("Create Brush from Selection", self._create_brush_from_selection))
        if selection_group:
            groups.append(selection_group)

        if not groups:
            return

        menu = QMenu(self)
        handlers: dict = {}
        for i, group in enumerate(groups):
            if i > 0:
                menu.addSeparator()
            for label, handler in group:
                action = menu.addAction(label)
                handlers[action] = handler

        chosen = menu.exec(event.globalPos())
        if chosen in handlers:
            handlers[chosen]()

    def _commit_edit(self, before: np.ndarray, after: np.ndarray) -> None:
        if self._on_stroke_committed and not np.array_equal(before, after):
            self._on_stroke_committed(self.layers.active_index, before, after)

    def _copy_selection(self) -> None:
        if self.selection is None or self.selection.is_empty():
            return
        x0, y0, x1, y1 = self.selection.bounds()
        layer = self.layers.active
        region = layer.pixels[y0:y1, x0:x1].copy()
        mask = self.selection.mask[y0:y1, x0:x1].copy()
        region[~mask] = 0
        self._clipboard = {"pixels": region, "mask": mask, "origin": (x0, y0)}

    def _cut_selection(self) -> None:
        if self.selection is None or self.selection.is_empty():
            return
        self._copy_selection()
        layer = self.layers.active
        before = layer.pixels.copy()
        layer.pixels[self.selection.mask] = 0
        self.mark_dirty()
        self._commit_edit(before, layer.pixels.copy())

    def _paste_clipboard(self) -> None:
        if self._clipboard is None:
            return
        layer = self.layers.active
        before = layer.pixels.copy()
        clip_pixels, clip_mask = self._clipboard["pixels"], self._clipboard["mask"]
        h, w = clip_pixels.shape[:2]
        # Paste back at the exact spot it was copied/cut from by default (like
        # every other editor's plain Paste), rather than re-centering on wherever
        # the mouse happens to be hovering (which can even be nowhere, e.g. right
        # after using the right-click context menu).
        x0, y0 = self._clipboard["origin"]

        lh, lw = layer.pixels.shape[:2]
        dst_x0, dst_y0 = max(0, x0), max(0, y0)
        dst_x1, dst_y1 = min(lw, x0 + w), min(lh, y0 + h)
        if dst_x1 <= dst_x0 or dst_y1 <= dst_y0:
            return
        src_x0, src_y0 = dst_x0 - x0, dst_y0 - y0
        src_x1, src_y1 = src_x0 + (dst_x1 - dst_x0), src_y0 + (dst_y1 - dst_y0)

        src_pixels = clip_pixels[src_y0:src_y1, src_x0:src_x1]
        src_mask = clip_mask[src_y0:src_y1, src_x0:src_x1]
        dst = layer.pixels[dst_y0:dst_y1, dst_x0:dst_x1]
        dst[src_mask] = src_pixels[src_mask]

        new_mask = np.zeros((lh, lw), dtype=bool)
        new_mask[dst_y0:dst_y1, dst_x0:dst_x1] = src_mask
        self.selection = Selection(mask=new_mask)
        self._lasso_points = None
        self.mark_dirty()
        self._commit_edit(before, layer.pixels.copy())

    def _flip_selection(self, axis: int) -> None:
        if self.selection is None or self.selection.is_empty():
            return
        layer = self.layers.active
        before = layer.pixels.copy()
        x0, y0, x1, y1 = self.selection.bounds()
        region = layer.pixels[y0:y1, x0:x1]
        mask = self.selection.mask[y0:y1, x0:x1]
        # np.flip returns a VIEW aliasing `region`'s memory, so the source must be
        # copied before `region` is mutated below, or the flipped read would see
        # already-cleared data.
        flipped_region = np.flip(region, axis=axis).copy()
        flipped_mask = np.flip(mask, axis=axis).copy()

        region[mask] = 0
        region[flipped_mask] = flipped_region[flipped_mask]
        self.selection.mask[y0:y1, x0:x1] = flipped_mask
        self._lasso_points = None
        self.mark_dirty()
        self._commit_edit(before, layer.pixels.copy())

    def _rotate_selection_90(self) -> None:
        if self.selection is None or self.selection.is_empty():
            return
        layer = self.layers.active
        before = layer.pixels.copy()
        x0, y0, x1, y1 = self.selection.bounds()
        region = layer.pixels[y0:y1, x0:x1].copy()
        mask = self.selection.mask[y0:y1, x0:x1].copy()
        region[~mask] = 0

        # Exact, lossless 90-degree rotation (k=-1 is clockwise for np.rot90),
        # rather than the general RotSprite path, which is approximate.
        rotated_region = np.rot90(region, k=-1)
        rotated_mask = np.rot90(mask, k=-1)

        layer.pixels[y0:y1, x0:x1][mask] = 0

        rh, rw = rotated_region.shape[:2]
        cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        nx0, ny0 = int(round(cx - rw / 2.0)), int(round(cy - rh / 2.0))

        lh, lw = layer.pixels.shape[:2]
        dst_x0, dst_y0 = max(0, nx0), max(0, ny0)
        dst_x1, dst_y1 = min(lw, nx0 + rw), min(lh, ny0 + rh)
        if dst_x1 > dst_x0 and dst_y1 > dst_y0:
            src_x0, src_y0 = dst_x0 - nx0, dst_y0 - ny0
            src_x1, src_y1 = src_x0 + (dst_x1 - dst_x0), src_y0 + (dst_y1 - dst_y0)
            src_region = rotated_region[src_y0:src_y1, src_x0:src_x1]
            src_mask = rotated_mask[src_y0:src_y1, src_x0:src_x1]
            dst = layer.pixels[dst_y0:dst_y1, dst_x0:dst_x1]
            dst[src_mask] = src_region[src_mask]

            full_mask = np.zeros((lh, lw), dtype=bool)
            full_mask[dst_y0:dst_y1, dst_x0:dst_x1] = src_mask
            self.selection = Selection(mask=full_mask)
            self._lasso_points = None
        self.mark_dirty()
        self._commit_edit(before, layer.pixels.copy())

    def _create_brush_from_selection(self) -> None:
        if self.selection is None or self.selection.is_empty():
            return
        x0, y0, x1, y1 = self.selection.bounds()
        mask = self.selection.mask[y0:y1, x0:x1]
        matrix = (mask.astype(np.uint8)) * 255
        brush = Brush(name=f"Selection {matrix.shape[1]}x{matrix.shape[0]}", matrix=matrix)
        idx = self.brush_manager.add(brush)
        self.brush_manager.active_index = idx
        self.set_tool(ToolKind.BRUSH)
        self.brush_created.emit()

    def set_tool(self, kind: ToolKind) -> None:
        self.active_tool_kind = kind
        self.active_tool = create_tool(kind)
        self._update_tool_label_text()

    def set_stroke_commit_callback(self, callback) -> None:
        self._on_stroke_committed = callback

    def fit_to_window(self) -> None:
        margin = 0.9
        zx = self.width() / max(self.layers.width, 1) * margin
        zy = self.height() / max(self.layers.height, 1) * margin
        self.zoom = float(np.clip(min(zx, zy), 0.25, 64.0))
        self.pan = QPointF(0.0, 0.0)
        self.zoom_changed.emit(self.zoom)
        self.update()
