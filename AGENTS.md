# AGENTS.md — OpenPixel

Guidance for AI agents (and contributors) working in this repository or
driving OpenPixel through its MCP server.

## What this project is

OpenPixel is a PySide6/PyOpenGL pixel-art editor. All application code lives
under `app/`; only project-level config and docs
(`requirements.txt`, this file, `README.md`, `.claude/`) sit at the repo
root. There are two ways to work with it:

1. **As a desktop app** (`app/main.py`) — a human-facing Qt GUI.
2. **As an MCP server** (`app/mcp_server.py`) — headless tools that let an
   agent create and edit pixel art programmatically. This is almost
   certainly what you want if you're an agent reading this file.

The GUI and the MCP server are two thin layers over the same core engine
modules. Neither layer duplicates editing logic — if you need a capability
that doesn't exist yet, add it to the relevant core module first (see
"Architecture" below), then expose it from whichever layer needs it. Logic
shared by *both* layers (not editing logic itself, but plumbing like image
codec conversion or thumbnail rendering) belongs in `app/utils/`, not
duplicated in both `main.py` and `mcp_server.py` — that duplication used to
exist and was consolidated into `app/utils/image_io.py` and
`app/utils/thumbnails.py`; keep it that way.

## Running the MCP server

```
python app/mcp_server.py
```

This starts a stdio MCP server (see `mcp.run()` at the bottom of the file).
Register it with your MCP client the normal way (e.g. `claude mcp add
openpixel -- python /path/to/app/mcp_server.py`, or the equivalent for your
client). No GUI window opens — Qt is used only for image codecs (PNG/JPEG/
WebP encode-decode), not for any window.

**If you're an agent using the OpenPixel MCP tools to create pixel art, read
both skills before you start:**

- `.claude/skills/pixel-art-craft/SKILL.md` — how a pixel artist actually
  *thinks*: silhouette-first, value before color, palette planning,
  readability, animation sequencing. This decides *what* to draw.
- `.claude/skills/pixel-art-mcp/SKILL.md` — the tool-call workflow: canvas
  sizing, checking your work with `get_screenshot`/`validate_render`, sprite
  sheet conventions, region operations, common pitfalls. This decides *how*
  to execute it in OpenPixel.

This file (AGENTS.md) covers the codebase itself, not the craft of using it.

## Architecture

All paths below are relative to `app/`.

Core engine modules (pure logic, no Qt widgets, safe to import headlessly):

- `layers.py` — `LayerStack`/`Layer`: RGBA `uint8` numpy buffers, opacity,
  blend modes (Normal/Multiply/Screen), compositing.
- `palette.py` — `Palette`: indexed colors, `.gpl`/`.hex` import/export,
  nearest-color matching, quantization.
- `brushes.py` — `Brush`/`BrushManager`: coverage-mask brushes (solid,
  circle, Bayer-matrix dither), `.pxbrush` JSON format, `stamp()`/`erase()`.
- `tools.py` — stateless drawing primitives (`bresenham_line`,
  `remove_diagonal_doubles`, `flood_fill`, `eyedropper`) plus the `Tool`
  class hierarchy the GUI's canvas drives during a mouse stroke. Prefer the
  primitives directly when scripting; the `Tool` classes exist for the GUI's
  begin/move/end stroke lifecycle.
- `filters.py` — pixel-art filters: palette quantize, outline generator,
  color-ramp shading, auto double-pixel cleanup, drop shadow, box blur
  (alpha-premultiplied so transparent neighbors don't darken edges),
  sharpen, brightness/contrast, invert, grayscale.
- `transform.py` — RotSprite-style nearest-neighbor scale/rotate for
  selections.
- `sprite_sheet.py` — `SpriteSheetInfo`: frame grid geometry, plus
  `detect_sprite_sheet_grid()`, a best-effort heuristic that finds frame
  boundaries from background/transparent gutters. It is *not* guaranteed
  correct — see "Sprite sheet detection is a heuristic" below.
- `project_io.py` — `.opxproj` save/load (a zip of a JSON manifest + one PNG
  per layer).
- `utils/image_io.py` — QImage↔numpy conversion, PNG/JPEG/WebP save/load, and
  sprite-sheet metadata embed/read. Shared by `main.py` and `mcp_server.py`;
  needs a live `QGuiApplication`/`QApplication` to work (see "Known sharp
  edges" below), but doesn't need a display or window.
- `utils/thumbnails.py` — QPixmap thumbnail rendering for brush/layer
  previews, shared by the GUI's dock panels.

GUI-only layers (Qt, do not import these from headless code):

- `canvas.py` — `PixelCanvas`, a `QOpenGLWidget` handling rendering, mouse
  input, pan/zoom, the brush-preview overlay, and the animation playback
  view. This is the only file that touches OpenGL.
- `main.py` — `MainWindow` and all dock/dialog widgets. Owns the
  `QUndoStack`; core modules have no undo concept of their own.
- `icons.py` — loads the SVG files in `icons/` as `QIcon`s.

`mcp_server.py` depends only on the core engine modules (including
`utils/image_io.py`, which needs `PySide6.QtGui` for `QImage` codec calls) —
it deliberately does not import `main.py` or `canvas.py`, so it never needs a
display or OpenGL context.

## Conventions

- **Color** is always `(r, g, b, a)`, `uint8`, 0–255. Fully transparent is
  `(0, 0, 0, 0)` — filters and the eraser rely on that exact zeroing, not
  just `a == 0`, so don't leave stale RGB in transparent pixels.
- **Coordinates**: `(0, 0)` is the top-left pixel; `x` is column, `y` is row.
  Array indexing is `pixels[y, x]`, not `pixels[x, y]` — this trips people up
  constantly, double check it in new code.
- **Layer order**: `LayerStack.layers[0]` is the bottom-most layer;
  `composite()` blends forward through the list, so the *last* layer ends up
  on top. The GUI's layer panel deliberately displays the list top-most-first
  (reversed) to match user expectation — don't "fix" that reversal, it's
  intentional (see `LayerStackPanel.refresh`).
- **New features go in the core module first.** If a filter, tool, or
  transform doesn't exist as a plain-numpy function in `tools.py`/
  `filters.py`/`transform.py`, add it there — don't implement editing logic
  directly inside `canvas.py`'s mouse handlers or an MCP tool function. Both
  layers should stay thin wrappers.
- **No comments explaining what code does** — name things clearly instead.
  Comments here exist only for non-obvious *why* (see the alpha-premultiply
  note in `box_blur`, or the layer-order reversal note above, as examples of
  the bar).

## Known sharp edges

- **Sprite sheet detection is a heuristic, not ground truth.**
  `detect_sprite_sheet_grid()` guesses frame boundaries from background
  gutters and can misjudge sheets where sprites overlap their frame edges
  inconsistently. When you (or an agent) create a sheet with
  `set_sprite_sheet`/OpenPixel's PNG export, the exact layout gets embedded
  in the PNG as a `openpixel:sprite_sheet` text chunk — always prefer reading
  that over re-detecting (`import_sprite_sheet_file` already does this
  automatically). Only fall back to detection for sheets that didn't come
  from OpenPixel.
- **Qt needs a live `QGuiApplication`/`QApplication`** before any
  `QImage`/`QBuffer` call works reliably (image format plugin loading). Both
  `main.py` and `mcp_server.py` create one at import/startup time — if you
  write a new standalone script that touches `QImage`, do the same
  (`QGuiApplication.instance() or QGuiApplication(sys.argv[:1])`).
- **Undo lives in the GUI only** (`main.py`'s `QUndoStack` + `PaintCommand`).
  The MCP server has no undo — each tool call mutates the session's layers
  directly. If you're scripting a multi-step edit and might want to back out,
  keep your own copy of the pixels you're about to overwrite, or work on a
  duplicate layer.
- **`np.flip()` and basic-slice views return views, not copies.** This bit us
  once already (a flip-selection bug where the source was read after being
  zeroed, because both aliased the same memory). Copy before you mutate
  whenever you're about to read from data that shares memory with what
  you're writing to.

## Testing

There's no permanent test suite — verification during development has been
done with disposable scripts (`_test_*.py`) written, run, and deleted in the
same session. `.gitignore` already excludes `_test_*.py` so these never get
committed. Follow the same pattern for anything new: write a small numpy-only
script that asserts against known outputs, run it, delete it.
