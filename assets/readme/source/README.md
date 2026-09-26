# Editable hero composition

`hero-layout.svg` keeps the text, layout, and named animation layers editable.
It references the repository's existing image files through relative paths.
The README displays `../hero.gif`; `../hero.png` is the complete static fallback.

Motion timing and layer IDs are stored in `../hero-motion.json`.
For raster export, resolve the referenced images before rendering the SVG.
Keep them as local references in this source; do not publish this layout as the README image.
