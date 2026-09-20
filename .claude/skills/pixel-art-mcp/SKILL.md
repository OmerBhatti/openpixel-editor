---
name: pixel-art-mcp
description: Use when creating or editing pixel art through the OpenPixel MCP server (tools like create_canvas, set_pixel, draw_line, flood_fill_area, apply_filter, get_screenshot, validate_render). Covers canvas sizing, palette discipline, checking your work, sprite sheets, regions/transforms, and common mistakes specific to pixel art (as opposed to general raster graphics).
---

# Creating pixel art with OpenPixel

OpenPixel's MCP server gives you low-level drawing primitives (`set_pixel`,
`draw_line`, `draw_rect`, `draw_circle`, `flood_fill_area`, `stamp_brush`)
plus layers, filters, palettes, sprite sheets, rectangular region operations
(copy/paste/flip/rotate/transform), and ways to inspect your own output
(`get_screenshot`, `validate_render`). It does not give you taste — the tools
will happily let you produce something that looks like a raster mess instead
of pixel art. This skill is about using them like a pixel artist would; see
**`pixel-art-craft`** for the actual design thinking (silhouette, value,
palette planning, readability) that should drive *what* you draw before you
reach for any of these tools.

## Start every session the same way

1. `create_canvas(width, height, background)` — **keep it small**. Pixel art
   reads as "pixel art" because every pixel is a deliberate decision at a
   size where that's tractable. 16x16 to 64x64 is the normal range for a
   single sprite; go up to 128x128 only for detailed scenes. If you catch
   yourself wanting 256x256+, you're probably drawing a regular raster image
   with extra steps, not pixel art.
2. Decide your palette *before* you start drawing, and set it with
   `set_palette`. A real limited palette (8-32 colors) is what makes shading
   read as intentional rather than muddy. Don't add a new one-off color every
   time you want to shade something — reuse or extend the palette
   deliberately, then `apply_filter(name="quantize")` if you drift.
3. If you're building an animation or a set of related frames, decide the
   frame size and grid now and call `set_sprite_sheet` up front, not after
   the fact.

## The core loop: draw, then look, then validate

**Call `get_screenshot` after every meaningful batch of edits** — prefer it
over `get_image` for anything smaller than ~64px, since it upscales with
nearest-neighbor so you can actually see individual pixels instead of
squinting at a postage stamp; pass `show_frame_markers=true` (the default)
when working on a sprite sheet so frame boundaries are visible, and
`show_pixel_grid=true` when you need to reason about exact pixel positions.
You cannot reliably judge pixel-level placement, silhouette readability, or
color balance from coordinates alone — you have to look. Treat this the same
way you'd treat re-reading a diff before deciding it's correct. A good
rhythm:

1. Make one coherent chunk of progress (e.g. "block in the base silhouette",
   "add the shading pass", "add the outline").
2. `get_screenshot` and actually look at it.
3. `validate_render` for a cheap, non-visual sanity check — it reports
   whether the layer/frame is unexpectedly empty, the tight bounding box of
   actual content (catches "I meant to draw in the center but it's offset"),
   unique color count, and (if you've set a palette) what fraction of pixels
   actually match it, so drift is caught immediately rather than at the end.
   For a sprite sheet it breaks this down per frame — a blank or off-palette
   frame in the middle of an animation is easy to miss visually but shows up
   immediately here.
4. Decide what's wrong before making more changes — don't chain five
   speculative edits without checking in between, because pixel art is
   unforgiving of compounding small errors (an off-by-one at 16x16 is a much
   bigger fraction of the image than at 1024x1024).

For sprite sheets, use `get_frame_image(col, row)` (a plain, non-upscaled
crop of exactly one frame) to inspect one frame in isolation when checking
animation consistency frame-to-frame, not just the whole sheet at once.

## Drawing technique

- **Outlines and diagonals**: use `draw_line` with `pixel_perfect=true`
  (the default) for clean 1px diagonals — this removes the "staircase double
  pixel" that a naive line leaves, which is the single most obvious tell of
  non-pixel-art-aware line drawing.
- **Fills**: use `flood_fill_area` for solid regions instead of drawing them
  pixel-by-pixel or as a filled rect over a busy area — it respects the
  existing silhouette and tolerance, a rect will not.
- **Textured shading**: `stamp_brush` with `brush="dither:2"`/`"dither:4"`/
  `"dither:8"` applies an ordered Bayer dither pattern at 50% coverage — a
  quick way to get a textured mid-tone between two palette colors without
  hand-placing a checkerboard.
- **Symmetry**: use `copy_region` to grab one half of a symmetric sprite (most
  characters, most game objects), then `paste_region` combined with
  `flip_region` on the pasted copy to mirror it into place — this guarantees
  exact symmetry, unlike eyeballing the second half pixel-by-pixel, which is
  never quite symmetric and it shows.
- **Moving/duplicating work**: `copy_region`/`paste_region` for relocating a
  finished piece (e.g. building one tile and stamping copies), `flip_region`/
  `rotate_region_90` for mirrored or 4-directional variants, and
  `transform_region` (RotSprite-style) only when you need a non-90-degree
  rotation or a scale change — it's lossier than the exact 90-degree rotation,
  so don't reach for it by default.
- **Anti-aliasing**: don't hand-place soft/blended edge pixels expecting them
  to look like AA at pixel-art scale; a 1-2px selective outline
  (`apply_filter(name="outline")`) or careful color-ramp shading
  (`apply_filter(name="ramp_shift")`) reads better than partial-alpha edges.
- **Shading**: prefer a small explicit color ramp (light → dark, in your
  palette) with hard bands, over `brightness_contrast` or `blur`, which
  produce continuous-tone results that fight the limited-palette look. Save
  `blur`/`sharpen` for pre-processing reference material you're about to
  quantize down, not for final pixel-art shading.

## Layers

Use layers the way you would in any editor — a base/silhouette layer, a
shading layer, a line-art layer, kept separate until you're happy, then treat
`get_canvas_info` as your layer inventory before deciding what to flatten or
adjust. Blend modes (`Multiply` for shading over a base color, `Screen` for
glow/highlight effects) are cheap ways to get lighting effects without
manually computing blended colors pixel-by-pixel. `duplicate_layer` before a
risky edit (there's no undo over MCP — see the note below) is cheaper than
redrawing from scratch if it goes wrong; `move_layer` reorders the stack, and
`resize_canvas` grows/crops every layer from the top-left corner if you
misjudged the canvas size after starting.

Note: **there is no undo over MCP.** Every tool call mutates the session's
layers directly and permanently. If you're about to try something you might
want to back out of, `duplicate_layer` first and work on the copy.

## Sprite sheets

- Frame order is row-major: frame index 0 is `(col=0, row=0)`, then left to
  right across the row, then down to the next row.
- If you're building the sheet from scratch, call `set_sprite_sheet` once you
  know the grid, then address individual frames with `get_frame_image`;
  drawing tools still operate on the shared canvas, so compute each frame's
  pixel offset yourself (`col * (frame_width + padding)`,
  `row * (frame_height + padding)`) when placing content into a specific
  frame with `set_pixel`/`draw_line`/etc.
- If you're editing a sheet imported via `import_sprite_sheet_file`, check
  `get_sprite_sheet_info` first — if the file came from OpenPixel's own PNG
  export the layout is exact (read from embedded metadata); if it came from
  elsewhere, the layout is a best-effort guess and you should sanity-check a
  couple of frames with `get_frame_image` before assuming the grid is right.

## Common mistakes to avoid

- Don't skip straight from "canvas created" to "call it done" without ever
  calling `get_screenshot` — you will not catch obvious errors (wrong colors,
  misplaced shapes, asymmetry) without looking, and `validate_render` doesn't
  substitute for looking, only for catching a narrower class of mistakes
  (empty layers, off-palette drift, unexpected bounding box).
- Don't use a new arbitrary color for every small shading decision — you'll
  end up with a "true color" image that happens to be small, not pixel art.
  Extend your palette deliberately instead.
- Don't forget `(0,0,0,0)` (fully zeroed, not just alpha=0) for erasing —
  leaving stale RGB in transparent pixels is harmless visually but will
  confuse filters and sprite-sheet gutter detection if the image is ever
  re-exported and re-analyzed.
- Don't build a huge canvas "to have room to work" — constrain yourself to
  the real target size from the start. Pixel art composition decisions don't
  transfer cleanly from a larger canvas down to a smaller one.
