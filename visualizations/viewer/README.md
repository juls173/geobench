# Guess viewer

A single-page browser for one dataset's results: the Street View image, the true
location against each model's guess on a map, and the model's own reasoning for
why it landed there.

```bash
python visualizations/viewer/build.py -d poland          # -> out/poland.html
python visualizations/viewer/build.py -d poland --open    # ...and open it
python visualizations/viewer/build.py -d japan --serve    # over http, if file:// is blocked
```

`build.py` picks up **every** run in `responses/` whose folder name carries that
dataset (`GPT-4o_poland_region_…`, `o3_poland_region_…`, …) and folds them into
one page. Narrow it with `-m o3 GPT-4o`. Point at other trees with `--responses`
and `--dataset-root`.

The output references dataset images by relative path rather than inlining them,
so the file stays under a megabyte and opens straight off disk. It reads
`results/detailed.csv`, `results/summary.json`, `output/<id>.txt` and — where the
provider recorded them — token counts and latency from `json/<id>.json`.
`out/` is gitignored; regenerate rather than commit.

## What's in it

**Browse** — one location at a time. Image (click or `f` to enlarge), a map with
the true point and the guess joined by the error line, the full reasoning with
the parsed `country/lat/lng` block called out, and a panel ranking what *every*
model said about this same location.

**Compare all** (`c`) — every model's guess on one map, numbered, with their
reasoning side by side.

**Overview** (`o`) — all guesses at once, a directional-bias plot showing where a
model systematically pulls, an error histogram, and a sortable model table.
Clicking any point jumps back into Browse at that location.

The left rail filters and sorts the 100 locations: worst first, best first, or by
**disagreement** (widest spread between models — usually the most interesting
images). The search box matches the reasoning text, so
`Białowieża` or `voivodeship` finds every location a model argued for it.
Clicking a histogram bar filters to that error band.

Keys: `←`/`→` location · `[`/`]` model · `c` compare · `o` overview · `f` image ·
`/` search. The URL hash tracks model and location, so a view is linkable.

## Notes

- Medians and averages follow `geobench.py` exactly — including its upper-middle
  median and its exclusion of refusals from the distance and score averages — so
  the header never disagrees with `summary.json`.
- Scores use the dataset's own GeoGuessr scale, printed in the footer. They are
  comparable **within** a dataset only.
- The basemap is OpenStreetMap, desaturated in CSS. Tiles and the web fonts need
  a network connection; without one the page still works and the location map
  falls back to a plain plot of the dataset bounds.

## Files

| | |
|---|---|
| `build.py` | collects runs + dataset, writes the HTML |
| `template.html` | the whole UI (CSS + JS); `build.py` substitutes `/*__DATA__*/` |
| `out/` | generated, gitignored |
