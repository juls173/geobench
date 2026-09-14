# Release-date timeline

An interactive scatter of every model GeoBench has scored against the day that model
shipped — the shape Epoch AI uses for its benchmark pages. Toggle the y-axis between
the GeoGuessr score (out of 5,000) and country accuracy, and the x-axis between the
two world image sets.

```bash
cd visualizations/timeline
python build_timeline.py --responses ../../responses
open timeline.html
```

`build_timeline.py` writes two files:

| File | What it is |
|---|---|
| `timeline_data.json` | the merged dataset — checked in, because `responses/` is not |
| `timeline.html` | the standalone page, `timeline_data.json` folded into `timeline.template.html` |

Edit `timeline.template.html` for anything about how the page looks or behaves, then
re-run the build. Editing `timeline.html` directly is pointless — the next build
overwrites it.

## Where the numbers come from

**Scores** merge two sources, both keyed by canonical model name:

1. The published leaderboard at <https://geobench.org>, scraped out of the JSON blob its
   JS bundle embeds.
2. Local `responses/*/results/summary.json` — runs that aren't on the leaderboard yet.

**Release dates** come from Epoch AI's [`benchmark_data.zip`](https://epoch.ai/data/benchmark_data.zip),
the same archive [`epoch-research/eci-public`](https://github.com/epoch-research/eci-public)
loads. Both tables in it get used:

- `model_metadata.csv` maps a model version (and its `model_group`) to a release date.
- `geobench_external.csv` is Epoch's own mirror of this leaderboard, and it names the
  exact model version behind each row. Where it has an opinion, it wins — a `model_group`
  can span versions with different dates (`gemini-1.5-flash-exp-0827` vs `-002`), so the
  group date alone is ambiguous.

Two models Epoch doesn't carry (`Qwen3-VL-235B`, `GLM-4.6V`) are dated from their vendor
announcements via `MANUAL_DATES`, and the page labels them so they're never mistaken for
Epoch's figures. Everything else resolves through `EPOCH_VERSIONS` / `MODELS`; at the end
of a run the script prints anything it could not map or date.

## Trend lines

Each of OpenAI, Google and Anthropic gets a fitted curve, with its rate of improvement
and R² on a card above the chart. Two shapes are offered:

- **Saturating** (default) fits a straight line in a *transformed* space and bends it
  back into the metric's own units — logit space for the bounded metrics, log space for
  miss distance. This is the shape Epoch's own ECI model assumes
  (`performance = sigmoid(discriminability × (capability − difficulty))`, `src/eci/fitting.py`).
  A logistic has no single slope, so the quoted pace is the curve's slope at that lab's
  most recent model.
- **Linear** is offered for comparison, and mostly to show where it breaks: fitted
  linearly, Anthropic's ACW score reaches a flawless 5,000 by August 2027.

On this data the two shapes fit about equally well — R² differs by roughly 0.03 — because
no lab has spent long enough near the ceiling to reveal the bend. They disagree about
extrapolation, not about the past, which is why curves are drawn only across the dates
each lab has actually shipped in.

R² is always computed in the metric's own units (predictions are back-transformed first),
so the two shapes are directly comparable. Fits pool every model a lab has on the image
set, small models alongside flagships, which is why `n` and R² are always on screen:
OpenAI's ACW score trend has R² ≈ 0.00, meaning no trend rather than a flat one.

## The saturation caveat

The GeoGuessr score decays exponentially in distance, so equal gains in accuracy buy
ever-smaller gains in points — the metric manufactures some of its own apparent
saturation. The **Median miss** axis is the same runs read as kilometres on a log scale,
and it is still falling by a roughly constant factor per year (on ACW: the field ×0.64/yr,
Anthropic ×0.41/yr). If you want to know whether progress has stopped, read that axis,
not the score.

## Two things the plot is careful about

**Scores don't cross image sets.** `calculate_score` scales the GeoGuessr formula by the
diagonal of the map's bounds, so an ACW point and a Photospheres point are not the same
achievement. The two sets are separate charts, never pooled — the same reason
`CLAUDE.md` warns that scores are only comparable within a dataset.

**One point per released model.** Runs of the same model at different reasoning efforts
(`Low` / `Medium` / `High` / `XHigh`), thinking budgets, with search, or under a
non-standard `prompt_mode` are *variants*: they share a release date and collapse to one
point, taking the best value on whichever axis is showing. Search and tuned-prompt runs
are labelled and separately filterable, because including them compares a tuned setup
against everyone else's standard one — Claude Fable 5's `geoguessr` prompt run scores
4,472 on ACW against 4,449 for the same model on the standard prompt. Because the two metrics can disagree about which run was
best, the score axis and the country axis may be reading different runs of the same
model — that's deliberate. Tick **Every reasoning-effort run** to see the whole spread.

## Adding a model

`models.py` needs no registry, but this script does, because it has to reconcile three
naming schemes. After a new run lands in `responses/`:

1. Add its display name to `MODELS` — `_m(raw_name, canonical, epoch_group, variant)`.
2. Re-run the build. If the script prints the model under *No release date found*, find
   its `model_group` in `model_metadata.csv` and fix the third argument, or pin the exact
   version in `EPOCH_VERSIONS`.

Unmapped names are reported rather than silently dropped, so a typo shows up as a missing
point with a line of output explaining it.
