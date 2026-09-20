# OpenPixel

A pixel art editor built with PySide6 (Qt6) and PyOpenGL, usable either as a
desktop app or headlessly through an MCP server so AI agents can create pixel
art programmatically.

## Features

- OpenGL canvas with nearest-neighbor rendering, smooth pan/zoom, a pixel
  grid, and a live brush-size preview outline
- Layers with opacity and blend modes (Normal / Multiply / Screen), with
  thumbnails and reordering
- Tools: Pencil (with pixel-perfect diagonal cleanup), Custom Brush (square,
  circle, ordered-dither presets, or a hand-drawn matrix), Eraser, Bucket
  Fill (with tolerance and sample-all-layers), Rectangle/Lasso selection,
  Eyedropper, Hand, Zoom
- Selection actions: copy/cut/paste, flip horizontal/vertical, rotate 90°,
  free transform (scale/rotate), and "create brush from selection"
- Indexed color palette panel with `.gpl`/`.hex` import/export
- Filters: palette quantize, outline generator, color-ramp shading,
  auto-clean stray diagonal pixels, drop shadow, blur, sharpen,
  brightness/contrast, invert, grayscale
- Sprite sheets: define a frame grid (linear/vertical/grid + padding) on a
  new canvas, or import an existing sheet with auto-detected frame geometry;
  an animation playback bar (play/pause/step/FPS) previews the frames
- Project files (`.opxproj`) preserving layers, palette, and sprite sheet
  layout; PNG/JPEG/WebP import; PNG export (with sprite-sheet geometry
  embedded for lossless re-import) and a sprite-sheet PNG exporter
- An MCP server (`mcp_server.py`) exposing the full editing engine as ~40
  tools — canvas/project/layer management, drawing primitives, brush
  stamping, rectangular region copy/paste/flip/rotate/transform, filters,
  palettes, sprite sheets, and ways for an agent to check its own work
  (an upscaled `get_screenshot` and a `validate_render` diagnostic report) —
  so an agent can create/edit pixel art entirely without the GUI

## Requirements

- Python 3.10+
- Windows, macOS, or Linux with a working OpenGL driver (only needed for the
  desktop app — the MCP server never opens a window or touches OpenGL)

## Setup

```bash
# from the repo root (the folder containing app/ and requirements.txt)
python -m venv venv

# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

pip install -r requirements.txt
```

## Running the desktop app

```bash
python app/main.py
```

## Running the MCP server

The MCP server lets an agent create and edit pixel art through tool calls —
canvas/project/layer management, drawing primitives, brush stamping, region
copy/paste/flip/rotate/transform, filters, palettes, sprite sheets, and tools
for the agent to inspect its own output (`get_screenshot`, `get_image`,
`get_frame_image`, `validate_render`). It talks over stdio and never opens a
GUI window.

```bash
python app/mcp_server.py
```

To register it with an MCP-capable client, point the client at this command
with your virtualenv active (so `python` resolves to the interpreter with
`mcp`/`PySide6`/`numpy` installed). For example, with Claude Code, from the
repo root:

```bash
claude mcp add openpixel -- python app/mcp_server.py
```

This repo also ships two ready-made, portable config files (no hardcoded
machine-specific paths, so they work after cloning on any machine):

- **`.mcp.json`** (project-scoped, auto-detected by Claude Code when you open
  this repo) — uses `${CLAUDE_PROJECT_DIR}` so the path resolves correctly
  regardless of where you cloned the repo.
- **`openpixel.mcp.json`** — the JSON body for `claude mcp add-json`, if you'd
  rather register it at the user/local scope instead:
  ```bash
  claude mcp add-json openpixel "$(cat openpixel.mcp.json)"
  ```
  Run this from the repo root — it uses a path relative to the current
  directory, not `${CLAUDE_PROJECT_DIR}` (that variable is only expanded for
  project-scoped `.mcp.json`).

Either way, `python` must resolve (PATH or an active virtualenv) to an
interpreter with this project's dependencies installed.

`.mcp.json` is git-ignored (not `openpixel.mcp.json`): if your client doesn't
expand `${CLAUDE_PROJECT_DIR}` at launch time, the server fails to start with
a "connection closed" error, and the working fix is to edit your local
`.mcp.json` to use a hardcoded absolute path to `app/mcp_server.py` instead —
which is machine-specific and shouldn't be committed.

Once connected, an agent should read both skills under `.claude/skills/`
before just calling tools ad hoc:

- [`pixel-art-craft`](.claude/skills/pixel-art-craft/SKILL.md) — how a pixel
  artist actually thinks (silhouette first, value before color, palette
  planning, readability, animation sequencing).
- [`pixel-art-mcp`](.claude/skills/pixel-art-mcp/SKILL.md) — the tool-call
  workflow for executing that in OpenPixel (canvas sizing, checking your work,
  sprite sheet conventions, region operations, common pitfalls).

See [`AGENTS.md`](AGENTS.md) for the codebase architecture if you're
extending either the app or the server.

## Project structure

Everything the app needs to run lives under `app/`; only project-level
config and docs (`requirements.txt`, `README.md`, `AGENTS.md`, `.claude/`)
stay at the repo root.

| File | Purpose |
|---|---|
| `app/main.py` | Desktop app entry point, main window, docks, dialogs |
| `app/canvas.py` | OpenGL viewport widget (rendering, input, playback view) |
| `app/layers.py` | Layer stack, compositing, blend modes |
| `app/tools.py` | Drawing primitives and the GUI tool-stroke state machine |
| `app/brushes.py` | Brush matrices, presets, `.pxbrush` format |
| `app/palette.py` | Indexed palettes, `.gpl`/`.hex` I/O, quantization |
| `app/filters.py` | Pixel-art filters |
| `app/transform.py` | Selection scale/rotate (RotSprite-style) |
| `app/sprite_sheet.py` | Sprite sheet layout + gutter-based auto-detection |
| `app/project_io.py` | `.opxproj` save/load |
| `app/icons.py` | SVG icon loading for the GUI |
| `app/mcp_server.py` | Headless MCP server exposing the engine as tools |
| `app/icons/` | SVG source icons |
| `app/utils/image_io.py` | Shared QImage↔numpy conversion, PNG/JPEG/WebP I/O, sprite-sheet metadata embedding — used by both `main.py` and `mcp_server.py` |
| `app/utils/thumbnails.py` | Shared QPixmap thumbnail renderers for brush/layer previews |

## Output folder

`File → Export PNG...` and `File → Save Project...` default to an `out/`
folder created next to the project (git-ignored).
