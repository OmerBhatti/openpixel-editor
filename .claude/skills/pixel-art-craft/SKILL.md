---
name: pixel-art-craft
description: Use when designing or evaluating pixel art (a sprite, icon, tileset, or animation) — not just operating the tools, but deciding what to actually draw. Covers how a pixel artist plans and sequences a piece: silhouette-first, value before color, palette planning, readability at target size, and a self-critique checklist. Pair with pixel-art-mcp for the tool-call mechanics of executing this process in OpenPixel.
---

# How a pixel artist thinks

Tool fluency (this skill's companion, `pixel-art-mcp`) doesn't produce good
pixel art by itself. What does is a design *process* — deciding what to draw
and in what order, before and while placing pixels. This skill is that
process.

## The core constraint everything else follows from

Pixel art is small on purpose. Every decision below exists because you have
few pixels and (usually) few colors, so nothing can be vague — a shape either
reads or it doesn't, a color either belongs to the palette or it doesn't.
Treat that scarcity as the design problem you're solving, not an obstacle
around it.

## Sequence: work in passes, not top-to-bottom

Don't render one pixel at a time in reading order. Work in these passes, each
one committing a decision the next pass depends on:

1. **Silhouette.** Block in the shape as one flat, solid color (or one color
   per major mass — head/body/limb). Check: does the shape read as what it's
   supposed to be with *zero* internal detail? If you can't tell a sword from
   a stick at this stage, no amount of shading fixes it — go back and change
   the shape.
2. **Value (light/dark), not color yet.** Decide where the light source is
   and block in 2-3 value levels (base, shadow, and maybe a highlight) using
   grayscale or placeholder colors. This is where the object starts reading
   as three-dimensional or as having material (metal vs. cloth vs. skin).
   Getting value right matters more than getting hue right — a correctly-lit
   sprite in the wrong hue still reads; a correctly-hued sprite with flat or
   confused lighting looks like a sticker.
3. **Color.** Now assign real colors from your palette to each value level.
   Keep the value relationships you already established — swapping in color
   shouldn't flatten the contrast you just built.
4. **Line art / outline (if the style uses one).** Add or clean up outlines
   last, once you know where shapes actually ended up, not first as a
   coloring-book template — outlining before you've finalized the silhouette
   just means re-drawing the outline later anyway.
5. **Cleanup pass.** Zoom to 100% (actual target size) and look for: stray
   pixels, jaggy lines that should be pixel-perfect diagonals, anti-aliasing
   halos that don't belong in a hard-edged style, and asymmetry that should
   be symmetric or vice versa.

Skipping straight to color-and-detail without the silhouette/value passes is
the single most common way pixel art ends up looking like a shrunken regular
illustration instead of pixel art.

## Design decisions to make explicitly, before you start

- **What's the read distance / display size?** A 16x16 inventory icon and a
  16x16 overworld character sprite have different jobs — an icon must read
  instantly and alone; a character sprite reads alongside a whole scene and
  can rely on animation and context. Decide this before choosing detail
  level, because it changes how much you can afford to simplify.
- **Where's the light source?** Pick one and stay consistent across every
  sprite in a set — inconsistent lighting is one of the fastest ways for a
  set of sprites to look like they don't belong together.
- **What's the palette, and why these colors?** A palette isn't just "colors
  I like" — plan a ramp (a light-to-dark sequence) per major material/surface
  so shading has somewhere to go. Reusing one ramp's colors across multiple
  objects (a shared "skin" ramp, a shared "metal" ramp) is what makes a set
  of sprites feel like one piece of art instead of unrelated stickers.
- **How much contrast can you afford?** Limited palettes need each color to
  pull weight. If two adjacent colors are close enough in value that you
  can't tell them apart when the piece is small, they're not doing separate
  jobs — merge them or push them further apart.

## Readability checklist (self-critique before calling it done)

Ask these, roughly in this order, and go back a pass if any answer is no:

1. Does the silhouette alone communicate the subject? (Cover the colored
   version mentally and imagine just the outline.)
2. Is the light source consistent with the rest of the set/scene?
3. Does each value level actually look distinct at the real display size —
   not just when you're zoomed in staring at individual pixels?
4. Are there any 1px orphan pixels, accidental gaps in outlines, or a
   diagonal line that's 2px thick in one place and 1px in another?
5. If this is a set (tileset, animation, matching UI icons): is the palette
   actually shared, or did a new color sneak in for just this one piece?
6. If this is an animation frame: does it stay "on model" with the other
   frames (same proportions, same palette, same silhouette language), or did
   the character subtly change shape between frames?

## Animation-specific thinking

For a walk/attack/idle cycle, plan the *poses* before drawing full detail on
any single frame:

- Rough in the silhouette of every key pose first (e.g. contact, passing,
  contact, passing for a walk cycle) so you can check the motion reads as a
  sequence before investing detail in any one frame.
- Keep the palette and proportions identical across frames — an animation
  where the character's color ramp or size drifts frame-to-frame reads as
  flickering/wobbling even if each individual frame looks fine in isolation.
- Exaggerate weight shifts and squash/stretch more than feels natural at
  first glance — subtlety that reads fine in a large illustration tends to
  disappear entirely at pixel-art scale and frame rate.

## When you're not sure if something is "pixel art enough"

A useful gut check: if you removed the pixel grid, would this still look
deliberate, or would it look like a low-resolution photo? Soft gradients,
partial-alpha edges everywhere, and continuous-tone shading are the tells of
the latter. Hard color bands, a limited palette used consistently, and clean
1px linework are the tells of the former.
