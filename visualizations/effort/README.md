# Reasoning effort vs score

Static matplotlib plots for every model that was run at more than one reasoning
effort. A run folder named `<Model> (<Effort>)_<dataset>_<timestamp>` is one rung of a
ladder; the script collects the rungs and draws them.

```bash
pip install -e ".[viz]"                     # matplotlib
python visualizations/effort/plot_effort.py # from the repo root
```

Output lands in `visualizations/effort/plots/`:

| File | What it shows |
|---|---|
| `summary_effort_delta.png` | **Start here.** Highest effort minus lowest for every ladder, with 95% CIs. |
| `overview_score_vs_effort.png` | All 14 ladders as small multiples, average score vs effort. |
| `<model>.png` | One model: average score, country accuracy and median miss vs effort. |

## Why the comparison is paired

Every rung of a ladder is scored on the same locations, so the runs are compared
**location by location** rather than by subtracting two averages. This matters more than
it sounds: a per-location score ranges over the whole 0–5,000 scale, and most of that
variance is "was this panorama easy?", which is identical across rungs and cancels
exactly when paired. The differences between efforts are ~100 points — small enough that
an unpaired comparison can't see them at all.

Error bars are ±1 standard error across locations for score and country accuracy, and a
2,000-sample bootstrap interval for the median miss, which has no closed-form standard
error. Fixed seed, so the plots are reproducible.

## What the plots say

**Mostly, more reasoning effort does not help.** Of the 14 model×dataset ladders, 11 show
a highest-vs-lowest difference that is not distinguishable from noise. Of the three that
clear their confidence interval, two are gains (GPT-5.6 Luna on ACW +344, GPT-5.5 on
Photospheres +278) and one is a loss (GPT-6 Astra on ACW −186).

The raw averages look more dramatic than that — Claude Opus 5 and GPT-6 Astra both
decline monotonically from Low to XHigh on ACW, and several models peak at Medium — but
those slopes sit inside their error bars. Read `summary_effort_delta.png` before
concluding a model gets worse when it thinks harder; on ACW only GPT-6 Astra actually
does, and only by about 4% of its score.

This is a plausible result rather than a surprising one: geolocation is largely a
recognition task, and extra deliberation gives a model more room to talk itself out of a
correct first impression.

## Scope

Only the `standard` prompt mode is included — a run with a tuned `prompt_mode` is a
different setup, not a different effort, and the script prints any it skips. Rungs are
taken from `responses/` directly, so the plots pick up new runs with no registry to
update; a model needs at least two efforts on the same dataset to appear.
