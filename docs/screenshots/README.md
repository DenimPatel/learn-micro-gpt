# Social preview

`og-image.svg` is the source; `og-image.png` is what GitHub's social preview
actually fetches, and it is the copy served at `/og-image.png` by the site.

The numbers in it are the **recorded** attention weights at step 200, position 3,
head 0, read from `traces/python/micro/attn.jsonl` — 0.394, 0.236, 0.197, 0.174.
Not a decorative pattern: the last bar being the *smallest* is the finding the
multi-head-attention concept is about, and the whole preview is that point.

Regenerate the PNG after editing the SVG:

```sh
rsvg-convert -w 1280 -h 640 docs/screenshots/og-image.svg -o docs/screenshots/og-image.png
cp docs/screenshots/og-image.png visualizer/public/og-image.png
```

`rsvg-convert` is in `librsvg` (`brew install librsvg`, `apt install librsvg2-bin`).
It is a manual step on purpose: taking a rasteriser as a build dependency to
produce one image would be a strange trade in a repository whose tooling has no
dependencies at all.

If the trace is regenerated and those weights move, update the SVG too — the
tests do not check an image, and a preview that quotes a stale number is exactly
the kind of thing this repository is about.
