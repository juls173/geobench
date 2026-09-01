# Guess viewer

A single-page browser for benchmark results: the Street View image, the true
location against each model's guess on a map, and the model's own reasoning for
why it landed there.

```bash
python visualizations/viewer/build.py            # every dataset -> out/geobench.html
python visualizations/viewer/build.py --open      # ...and open it
python visualizations/viewer/build.py -d poland   # one dataset -> out/poland.html
python visualizations/viewer/build.py --serve     # over http instead
```

With no `-d`, `build.py` sweeps every dataset that has runs and puts them all in
one file behind a dataset switcher (eleven datasets is ~3 MB). Name one or more
to narrow it: `-d poland japan`.

For each dataset it picks up **every** run in `responses/` whose folder name
carries that dataset (`GPT-4o_poland_region_…`, `o3_poland_region_…`, …). Narrow
the models with `-m o3 GPT-4o`. Point at other trees with `--responses` and
`--dataset-root`.

The output references dataset images by relative path rather than inlining them,
so the file stays small and opens straight off disk. It reads
`results/detailed.csv`, `results/summary.json`, `output/<id>.txt` and — where the
provider recorded them — token counts and latency from `json/<id>.json`.
`out/` is gitignored; regenerate rather than commit.

## Opening it

**On the machine with a desktop** — `--open`, or `xdg-open
visualizations/viewer/out/geobench.html`. No server needed: the images are
referenced by relative path and load fine over `file://`.

**Over SSH** — `--serve` starts a local server on 127.0.0.1 and prints the URL.
It roots itself high enough to cover both the page and `dataset/`, so forward the
port and open that URL on your own machine:

```bash
python visualizations/viewer/build.py --serve      # on the remote box
ssh -L 8020:127.0.0.1:8020 <host>                  # from your laptop
```

It binds to loopback only — the dataset is not exposed to the network.

## What's in it

**Browse** — one location at a time. Image (click or `f` to enlarge), a map with
the true point and the guess joined by the error line, the full reasoning with
the parsed `country/lat/lng` block called out, and a panel ranking what *every*
model said about this same location.

**Compare all** (`c`) — every model's guess on one map, numbered, with their
reasoning side by side.

**Overview** (`o`) — all guesses at once, a directional-bias plot showing where a
model systematically pulls, an error histogram, a sortable model table, and — with
more than one dataset — an **Across datasets** table ranking countries by
difficulty. Clicking any point or row jumps to it.

The left rail filters and sorts the 100 locations: worst first, best first, or by
**disagreement** (widest spread between models — usually the most interesting
images). The search box matches the reasoning text, so
`Białowieża` or `voivodeship` finds every location a model argued for it.
Clicking a histogram bar filters to that error band.

Keys: `←`/`→` location · `[`/`]` model · `,`/`.` dataset · `c` compare ·
`o` overview · `f` image · `/` search. The URL hash tracks dataset, model and
location, so a view is linkable.

## Notes

- Medians and averages follow `geobench.py` exactly — including its upper-middle
  median and its exclusion of refusals from the distance and score averages — so
  the header never disagrees with `summary.json`.
- Scores use the dataset's own GeoGuessr scale, printed in the footer. They are
  comparable **within** a dataset only — which is why the across-datasets table
  shows raw kilometres beside points and says so.
- Histogram bins are sized per model from the 92nd percentile of its errors.
  Poland lands on 50 km bins, Mongolia on 100 km; a fixed bin would collapse one
  of them into a single overflow bar.
- The basemap is OpenStreetMap, desaturated in CSS. Tiles and the web fonts need
  a network connection; without one the page still works and the location map
  falls back to a plain plot of the dataset bounds.

## Files

| | |
|---|---|
| `build.py` | collects runs + dataset, writes the HTML |
| `template.html` | the whole UI (CSS + JS); `build.py` substitutes `/*__DATA__*/` |
| `out/` | generated, gitignored |
