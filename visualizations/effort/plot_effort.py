"""Plot reasoning effort against score, for every model run at several efforts.

A run folder named ``<Model> (<Effort>)_<dataset>_<timestamp>`` is one rung of a
ladder; this collects the rungs and draws, per model:

* ``<model>_<dataset>.png``  -- avg score, country accuracy and median miss
  against effort, with error bars.
* ``overview_score_vs_effort.png`` -- every ladder as a small multiple.
* ``summary_effort_delta.png`` -- the headline: highest effort minus lowest,
  measured **per location** rather than by subtracting two averages.

The paired measurement matters. Every run in a ladder sees the same locations,
so comparing runs location-by-location cancels out the "was this panorama easy?"
variance that dominates an unpaired comparison -- the per-location score swings
over the whole 0-5000 range, and differences between efforts are ~100 points.
Subtracting two averages hides whether that difference is real; the paired
standard error shows it.

    python visualizations/effort/plot_effort.py                 # from the repo root
    python plot_effort.py --responses ../../responses           # from this directory

Needs matplotlib: ``pip install -e ".[viz]"``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent

# The ladder, in order. Anything else in the parentheses is not an effort level.
EFFORTS = ["Low", "Medium", "High", "XHigh"]
EFFORT_RANK = {e: i for i, e in enumerate(EFFORTS)}
RUN_RE = re.compile(r"^(?P<base>.*?)\s*\((?P<eff>Low|Medium|High|XHigh)\)\s*$")

# Same hues as the release-date timeline, so the two read as one family.
LAB_COLOR = {
    "Anthropic": "#1baf7a",
    "OpenAI": "#2a78d6",
    "Google": "#eb6834",
}
OTHER_COLOR = "#8a8d94"
INK, INK_2, INK_3 = "#16181c", "#4d525c", "#767a83"
GRID = "#e3e5ea"

DATASET_STYLE = {          # dataset -> (marker, linestyle, pretty name)
    "acw": ("o", "-", "ACW (100 locations)"),
    "photospheres": ("s", "--", "Photospheres (50 locations)"),
}


def style() -> None:
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK_2,
        "axes.titlecolor": INK,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": INK_3,
        "ytick.color": INK_3,
        "text.color": INK,
        "font.size": 10,
        "font.family": "DejaVu Sans",
        "legend.frameon": False,
        "figure.dpi": 150,
    })


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #

class Run:
    def __init__(self, folder: Path, summary: dict, base: str, effort: str):
        self.folder, self.summary = folder, summary
        self.base, self.effort = base, effort
        self.dataset = summary["test"]
        self.provider = summary.get("provider", "")
        self.n = int(summary.get("n", 0))
        self.avg_score = float(summary["average_score"])
        self.country = float(summary["country_success_rate"])
        self.median_km = summary.get("median_distance_km")
        self.refusal = float(summary.get("refusal_rate", 0.0))
        self.prompt_mode = (summary.get("prompt_mode") or "standard").strip()
        self._per_loc: dict[int, tuple[float, float, float]] | None = None
        self._cot: list[float] | None = None

    @property
    def per_location(self) -> dict[int, tuple[float, float, float]]:
        """location_id -> (score, country_correct as 0/1, distance_km)."""
        if self._per_loc is None:
            out: dict[int, tuple[float, float, float]] = {}
            path = self.folder / "results" / "detailed.csv"
            try:
                with path.open(newline="") as f:
                    for row in csv.DictReader(f):
                        try:
                            lid = int(row["location_id"])
                            score = float(row["score"] or 0)
                        except (KeyError, ValueError):
                            continue
                        cc = 1.0 if str(row.get("country_correct", "")).strip().lower() == "true" else 0.0
                        try:
                            dist = float(row.get("distance_km") or "nan")
                        except ValueError:
                            dist = float("nan")
                        out[lid] = (score, cc, dist)
            except OSError:
                pass
            self._per_loc = out
        return self._per_loc


    @property
    def cot(self) -> list[float]:
        """Chain-of-thought tokens per location, from the archived responses.

        Only the raw JSON records this -- summary.json has no token counts. The
        field is output_tokens_details.reasoning_tokens on OpenAI and
        .thinking_tokens on Anthropic; chat-completions shaped replies put the
        same thing under completion_tokens_details.
        """
        if self._cot is None:
            vals: list[float] = []
            for jf in sorted((self.folder / "json").glob("*.json")):
                try:
                    d = json.loads(jf.read_text())
                except (OSError, ValueError):
                    continue
                usage = d.get("usage") or {}
                det = usage.get("output_tokens_details") or usage.get("completion_tokens_details") or {}
                v = det.get("reasoning_tokens", det.get("thinking_tokens"))
                if isinstance(v, (int, float)):
                    vals.append(float(v))
            self._cot = vals
        return self._cot

    @property
    def cot_mean(self) -> float | None:
        return mean_sem(self.cot)[0] if self.cot else None


def load_runs(responses: Path) -> list[Run]:
    runs = []
    for summary_path in sorted(responses.glob("*/results/summary.json")):
        try:
            summary = json.loads(summary_path.read_text())
        except (OSError, ValueError):
            continue
        m = RUN_RE.match(summary.get("model", ""))
        if not m:
            continue
        runs.append(Run(summary_path.parent.parent, summary, m["base"], m["eff"]))
    return runs


def ladders(runs: list[Run]) -> dict[tuple[str, str], dict[str, Run]]:
    """(model, dataset) -> effort -> run, keeping only ladders with >= 2 rungs."""
    grouped: dict[tuple[str, str], dict[str, Run]] = defaultdict(dict)
    skipped_prompt = []
    for r in runs:
        if r.prompt_mode != "standard":
            # a tuned prompt is a different setup, not a different effort
            skipped_prompt.append(f"{r.base} ({r.effort}) [{r.prompt_mode}]")
            continue
        key = (r.base, r.dataset)
        prev = grouped[key].get(r.effort)
        if prev is None or r.folder.name > prev.folder.name:   # newest wins
            grouped[key][r.effort] = r
    if skipped_prompt:
        print(f"  skipped {len(skipped_prompt)} non-standard-prompt run(s): "
              + ", ".join(sorted(set(skipped_prompt))))
    return {k: v for k, v in grouped.items() if len(v) >= 2}


# --------------------------------------------------------------------------- #
# statistics
# --------------------------------------------------------------------------- #

def mean_sem(values: list[float]) -> tuple[float, float]:
    n = len(values)
    if n == 0:
        return float("nan"), float("nan")
    mean = sum(values) / n
    if n < 2:
        return mean, 0.0
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    return mean, math.sqrt(var / n)


def paired_delta(hi: Run, lo: Run, index: int) -> tuple[float, float, int]:
    """Mean of (hi - lo) over the locations both runs saw, with its SEM.

    index 0 = score, 1 = country-correct.
    """
    a, b = hi.per_location, lo.per_location
    shared = sorted(set(a) & set(b))
    diffs = [a[i][index] - b[i][index] for i in shared]
    mean, sem = mean_sem(diffs)
    return mean, sem, len(shared)


def median_ci(run: Run, reps: int = 2000, seed: int = 12345) -> tuple[float, float] | None:
    """95% bootstrap interval for the median miss distance.

    The median has no closed-form standard error, and leaving the panel without
    uncertainty invites reading a 40 km wobble as a real regression.
    """
    vals = [d for _, _, d in run.per_location.values() if not math.isnan(d)]
    n = len(vals)
    if n < 5:
        return None
    rng = random.Random(seed)
    meds = []
    for _ in range(reps):
        sample = sorted(vals[rng.randrange(n)] for _ in range(n))
        mid = n // 2
        meds.append(sample[mid] if n % 2 else (sample[mid - 1] + sample[mid]) / 2)
    meds.sort()
    return meds[int(0.025 * reps)], meds[min(int(0.975 * reps), reps - 1)]


def score_sem(run: Run) -> float:
    vals = [v[0] for v in run.per_location.values()]
    if not vals:
        return float("nan")
    return mean_sem(vals)[1]


def country_sem(run: Run) -> float:
    """Binomial standard error of the country rate."""
    n = run.n or len(run.per_location)
    if n <= 0:
        return float("nan")
    p = run.country
    return math.sqrt(max(p * (1 - p), 0) / n)


# --------------------------------------------------------------------------- #
# plots
# --------------------------------------------------------------------------- #

def header(fig, title: str, subtitle: str, title_size: float = 13.5) -> float:
    """Lay a left-aligned title + subtitle above the axes.

    Positions are measured in inches from the top and anchored with va="top",
    because suptitle's fractional y drifts against tight_layout and quietly
    lets the two lines overlap. Returns the fraction to reserve via rect.
    """
    h = fig.get_figheight()
    lines = subtitle.count("\n") + 1
    top_pad, gap, line_h = 0.24, 0.30, 0.17
    fig.text(0.005, 1 - top_pad / h, title, fontsize=title_size,
             ha="left", va="top", color=INK)
    fig.text(0.005, 1 - (top_pad + gap) / h, subtitle, fontsize=9,
             ha="left", va="top", color=INK_3, linespacing=1.45)
    return (top_pad + gap + lines * line_h + 0.16) / h


def colour_for(run: Run) -> str:
    return LAB_COLOR.get(run.provider, OTHER_COLOR)


def km_fmt(v, _pos=None):
    return f"{v:,.0f}"


def tok_fmt(v: float) -> str:
    """Token counts, compact enough to sit on a marker."""
    if v is None:
        return "—"
    if v >= 10000:
        return f"{v/1000:.0f}k"
    if v >= 1000:
        return f"{v/1000:.1f}k"
    return f"{v:.0f}"


def plot_model(model: str, by_dataset: dict[str, dict[str, Run]], out: Path) -> Path:
    """Three metrics against effort, one line per dataset."""
    fig, axes = plt.subplots(1, 4, figsize=(15.6, 4.1))
    any_run = next(iter(next(iter(by_dataset.values())).values()))
    colour = colour_for(any_run)

    panels = [
        ("Average score", "of 5,000", lambda r: r.avg_score, score_sem),
        ("Country accuracy", "% of locations", lambda r: r.country * 100,
         lambda r: country_sem(r) * 100),
        ("Median miss", "km (lower is better)", lambda r: r.median_km, None),
        ("Reasoning tokens", "avg per location", lambda r: r.cot_mean,
         lambda r: mean_sem(r.cot)[1] if r.cot else None),
    ]

    used_efforts: list[str] = []
    for ds, effs in by_dataset.items():
        used_efforts += list(effs)
    order = [e for e in EFFORTS if e in set(used_efforts)]
    xpos = {e: i for i, e in enumerate(order)}

    for ax, (title, ylab, getter, semf) in zip(axes, panels):
        for di, (ds, effs) in enumerate(sorted(by_dataset.items())):
            marker, ls, label = DATASET_STYLE.get(ds, ("^", ":", ds))
            rungs = sorted(effs.items(), key=lambda kv: EFFORT_RANK[kv[0]])
            xs = [xpos[e] for e, _ in rungs]
            ys = [getter(r) for _, r in rungs]
            if any(y is None for y in ys):
                continue
            if semf is None:                      # median: bootstrap interval
                cis = [median_ci(r) for _, r in rungs]
                if all(c is not None for c in cis):
                    yerr = [[y - c[0] for y, c in zip(ys, cis)],
                            [c[1] - y for y, c in zip(ys, cis)]]
                else:
                    yerr = None
            else:
                errs = [semf(r) for _, r in rungs]
                yerr = errs if all(e is not None and not math.isnan(e) for e in errs) else None

            if yerr is not None:
                ax.errorbar(xs, ys, yerr=yerr, color=colour, marker=marker, linestyle=ls,
                            linewidth=1.8, markersize=6, capsize=3, elinewidth=1,
                            markeredgecolor="white", markeredgewidth=1, label=label)
            else:
                ax.plot(xs, ys, color=colour, marker=marker, linestyle=ls, linewidth=1.8,
                        markersize=6, markeredgecolor="white", markeredgewidth=1, label=label)

            if title == "Reasoning tokens":
                # the two image sets sit almost on top of each other at low effort,
                # so push the second one's labels below the line
                dy, va = (10, "bottom") if di == 0 else (-11, "top")
                for xi, yi in zip(xs, ys):
                    ax.annotate(tok_fmt(yi), (xi, yi), textcoords="offset points",
                                xytext=(0, dy), ha="center", va=va,
                                fontsize=8.5, color=INK_2)
                continue

            # mark the winning rung
            best = max(range(len(ys)), key=lambda i: ys[i]) if title != "Median miss" \
                else min(range(len(ys)), key=lambda i: ys[i])
            ax.annotate(f"{ys[best]:,.0f}" if title != "Country accuracy" else f"{ys[best]:.0f}%",
                        (xs[best], ys[best]), textcoords="offset points", xytext=(0, 11),
                        ha="center", fontsize=8.5, color=INK, fontweight="bold")

        ax.set_title(title, fontsize=11, pad=9, loc="left")
        ax.set_ylabel(ylab, fontsize=9)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(order)
        ax.set_xlim(-0.35, len(order) - 0.65)
        ax.margins(y=0.22)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.grid(axis="x", visible=False)
        if title in ("Median miss", "Reasoning tokens"):
            ax.yaxis.set_major_formatter(FuncFormatter(km_fmt))

    axes[0].legend(loc="best", fontsize=8.5)
    top = header(fig, f"{model} — does more reasoning effort help?",
                 "Score and country: ±1 standard error across locations. "
                 "Median miss: 95% bootstrap interval. Reasoning tokens are the "
                 "model's own chain of thought,\naveraged per location. "
                 "Scores compare only within one image set.")
    fig.tight_layout(rect=(0, 0, 1, 1 - top))

    path = out / f"{slug(model)}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_overview(lads: dict[tuple[str, str], dict[str, Run]], out: Path) -> Path:
    items = sorted(lads.items(), key=lambda kv: (kv[0][1], kv[0][0]))
    cols = 4
    rows = math.ceil(len(items) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(3.15 * cols, 2.55 * rows), squeeze=False)

    for ax, ((model, ds), effs) in zip([a for row in axes for a in row], items):
        rungs = sorted(effs.items(), key=lambda kv: EFFORT_RANK[kv[0]])
        run0 = rungs[0][1]
        colour = colour_for(run0)
        xs = list(range(len(rungs)))
        ys = [r.avg_score for _, r in rungs]
        errs = [score_sem(r) for _, r in rungs]
        ax.errorbar(xs, ys, yerr=errs, color=colour, marker="o", linewidth=1.8,
                    markersize=5, capsize=2.5, elinewidth=0.9,
                    markeredgecolor="white", markeredgewidth=1)
        best = max(xs, key=lambda i: ys[i])
        ax.plot([xs[best]], [ys[best]], marker="o", markersize=10, mfc="none",
                mec=colour, mew=1.6)
        # how much thinking each rung actually bought
        for xi, (_, r) in zip(xs, rungs):
            ax.annotate(tok_fmt(r.cot_mean), (xi, 0), xycoords=("data", "axes fraction"),
                        textcoords="offset points", xytext=(0, 4), ha="center",
                        fontsize=7.5, color=INK_3)
        ax.set_title(f"{model}\n{DATASET_STYLE.get(ds, ('', '', ds))[2]}",
                     fontsize=9.5, loc="left", pad=6)
        ax.set_xticks(xs)
        ax.set_xticklabels([e for e, _ in rungs], fontsize=8.5)
        ax.set_xlim(-0.3, len(xs) - 0.7)
        ax.margins(y=0.28)
        ax.tick_params(labelsize=8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.grid(axis="x", visible=False)

    for ax in [a for row in axes for a in row][len(items):]:
        ax.set_visible(False)

    top = header(fig, "Average score against reasoning effort",
                 "Ring marks the best rung; error bars ±1 SE across locations. "
                 "Grey figures are average chain-of-thought tokens per location.\n"
                 "The rungs sit well inside each other's error bars — "
                 "see summary_effort_delta.png.", title_size=14)
    fig.tight_layout(rect=(0, 0, 1, 1 - top))
    path = out / "overview_score_vs_effort.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_summary(lads: dict[tuple[str, str], dict[str, Run]], out: Path) -> Path:
    """Highest effort minus lowest, measured per location."""
    entries = []
    for (model, ds), effs in lads.items():
        rungs = sorted(effs.items(), key=lambda kv: EFFORT_RANK[kv[0]])
        lo, hi = rungs[0][1], rungs[-1][1]
        delta, sem, n = paired_delta(hi, lo, 0)
        if n == 0:
            continue
        entries.append({
            "label": f"{model}  ·  {ds}",
            "lo": rungs[0][0], "hi": rungs[-1][0],
            "delta": delta, "ci": 1.96 * sem, "n": n,
            "colour": colour_for(hi),
            "tok_lo": lo.cot_mean, "tok_hi": hi.cot_mean,
        })
    entries.sort(key=lambda e: e["delta"])

    fig, ax = plt.subplots(figsize=(9.4, 0.46 * len(entries) + 2.1))
    ys = range(len(entries))
    ax.barh(list(ys), [e["delta"] for e in entries],
            xerr=[e["ci"] for e in entries],
            color=[e["colour"] for e in entries], alpha=0.9, height=0.62,
            error_kw=dict(ecolor=INK_3, elinewidth=1, capsize=3))
    ax.axvline(0, color=INK_2, linewidth=1.1)
    span = max(abs(e["delta"]) + e["ci"] for e in entries) * 1.30
    ax.set_xlim(-span, span)
    ax.set_yticks(list(ys))
    ax.set_yticklabels(
        [f'{e["label"]}   ({e["lo"]}→{e["hi"]}, {tok_fmt(e["tok_lo"])}→{tok_fmt(e["tok_hi"])} tok)'
         for e in entries], fontsize=9)
    ax.set_xlabel("Change in average score, highest effort minus lowest  "
                  "(paired per location, 95% CI)", fontsize=9.5)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)

    for i, e in enumerate(entries):
        tip = e["delta"] + (e["ci"] if e["delta"] >= 0 else -e["ci"])
        ax.annotate(f'{e["delta"]:+,.0f}', (tip, i),
                    textcoords="offset points", xytext=(7 if e["delta"] >= 0 else -7, 0),
                    va="center", ha="left" if e["delta"] >= 0 else "right",
                    fontsize=8.5, color=INK_2)

    top = header(fig, "Does the highest reasoning effort beat the lowest?",
                 "Whiskers are 95% confidence intervals; any that crosses zero means the\n"
                 "difference is not distinguishable from noise. Measured location by location,\n"
                 "so panorama difficulty cancels out. Token figures are the average chain of\n"
                 "thought at each end of the ladder.")
    fig.tight_layout(rect=(0, 0, 1, 1 - top))
    path = out / "summary_effort_delta.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def slug(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "-", name).strip("-").lower()


# --------------------------------------------------------------------------- #

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--responses", type=Path, default=REPO_ROOT / "responses")
    ap.add_argument("--out", type=Path, default=HERE / "plots")
    args = ap.parse_args()

    if not args.responses.is_dir():
        sys.exit(f"no responses directory at {args.responses}")

    style()
    args.out.mkdir(parents=True, exist_ok=True)

    runs = load_runs(args.responses)
    print(f"Found {len(runs)} runs with an effort level in the name.")
    lads = ladders(runs)
    if not lads:
        sys.exit("no model was run at more than one effort level")

    by_model: dict[str, dict[str, dict[str, Run]]] = defaultdict(dict)
    for (model, ds), effs in lads.items():
        by_model[model][ds] = effs

    print(f"\n{len(by_model)} models with a ladder, {len(lads)} model×dataset pairs:\n")
    written = []
    for model, by_ds in sorted(by_model.items()):
        for ds, effs in sorted(by_ds.items()):
            rungs = sorted(effs.items(), key=lambda kv: EFFORT_RANK[kv[0]])
            lo, hi = rungs[0][1], rungs[-1][1]
            d, sem, n = paired_delta(hi, lo, 0)
            sig = "significant" if abs(d) > 1.96 * sem else "within noise"
            print(f"  {model:17s} {ds:13s} "
                  + " ".join(f"{e}={r.avg_score:7.1f}/{tok_fmt(r.cot_mean):>5s}" for e, r in rungs)
                  + f"   {rungs[0][0]}→{rungs[-1][0]}: {d:+7.1f} ±{1.96*sem:5.1f} ({sig}, n={n})")
        written.append(plot_model(model, by_ds, args.out))

    written.append(plot_overview(lads, args.out))
    written.append(plot_summary(lads, args.out))

    print(f"\nWrote {len(written)} plots to {args.out}:")
    for p in written:
        print(f"  {p.name}")


if __name__ == "__main__":
    main()
