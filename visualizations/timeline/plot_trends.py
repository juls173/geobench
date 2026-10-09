"""Static plot: GeoBench score against release date, with a trend per lab.

Reads timeline_data.json (run build_timeline.py first) and writes
``trends_<dataset>.png`` next to this file.

Each point is one OpenAI, Google or Anthropic model at its best run; search and
tuned-prompt runs are left out so every lab is measured on the same setup. The
trend line is fitted in logit space against the 5,000-point ceiling -- the
sigmoid shape Epoch's ECI model assumes -- because a straight line on a bounded
score eventually predicts the impossible. Each line spans only the dates that
lab has actually shipped in.

    python plot_trends.py                # both image sets
    python plot_trends.py --dataset acw
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
CEILING = 5000.0

LABS = {"OpenAI": "#2a78d6", "Google": "#eb6834", "Anthropic": "#1baf7a"}
INK, INK_2, INK_3 = "#16181c", "#4d525c", "#80848c"

# "Professional player" average score, from the human baselines the
# geobench.org leaderboard publishes alongside the model results.
PRO_HUMAN = {"acw": 4100, "photospheres": 3700}

YEAR = 365.2425


def to_year(iso: str) -> float:
    d = dt.date.fromisoformat(iso)
    return d.year + (d.timetuple().tm_yday - 1) / YEAR


def to_date(y: float) -> dt.datetime:
    return dt.datetime(int(y), 1, 1) + dt.timedelta(days=(y - int(y)) * YEAR)


def logistic_fit(xs, ys):
    """OLS of logit(score / ceiling) on time -> (intercept, slope)."""
    zs = [math.log((y / CEILING) / (1 - y / CEILING)) for y in ys]
    mx, mz = sum(xs) / len(xs), sum(zs) / len(zs)
    b = (sum((x - mx) * (z - mz) for x, z in zip(xs, zs))
         / sum((x - mx) ** 2 for x in xs))
    return mz - b * mx, b


def best_scores(ds: dict, lab: str) -> list[tuple[float, float]]:
    out = []
    for m in ds["models"]:
        if m["org"] != lab:
            continue
        runs = [v for v in m["variants"] if not v.get("tools") and not v.get("prompt_tuned")]
        if runs:
            out.append((to_year(m["date"]), max(v["score"] for v in runs)))
    return sorted(out)


def plot(ds: dict, out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(10, 6))
    x_min, x_max = math.inf, -math.inf
    ends = []

    for lab, col in LABS.items():
        pts = best_scores(ds, lab)
        if len(pts) < 3:
            continue
        xs, ys = zip(*pts)
        x_min, x_max = min(x_min, xs[0]), max(x_max, xs[-1])
        a, b = logistic_fit(xs, ys)

        grid = [xs[0] + (xs[-1] - xs[0]) * i / 100 for i in range(101)]
        curve = [CEILING / (1 + math.exp(-(a + b * g))) for g in grid]

        ax.scatter([to_date(x) for x in xs], ys, s=40, color=col, alpha=0.55,
                   edgecolor="white", linewidth=1, zorder=3)
        ax.plot([to_date(g) for g in grid], curve, color=col, linewidth=2.8, zorder=4)

        p = curve[-1] / CEILING                       # slope of the fit at the latest model
        pace = CEILING * b * p * (1 - p)
        ends.append([curve[-1], lab, col, pace])

    # pro human baseline
    human = PRO_HUMAN.get(ds["id"])
    if human:
        # stop short of the lab labels so the line never runs through them
        ax.hlines(human, to_date(x_min - 0.06), to_date(x_max + 0.01), color=INK_2,
                  linewidth=1.4, linestyle=(0, (5, 4)), zorder=2)
        ax.text(to_date(x_min), human + 25, f"Pro human  {human:,}", color=INK_2,
                fontsize=10, ha="left", va="bottom")

    # lab names at the end of each line, spread apart if they crowd
    ends.sort()
    for i in range(1, len(ends)):
        ends[i][0] = max(ends[i][0], ends[i - 1][0] + 125)
    for y, lab, col, pace in ends:
        ax.text(to_date(x_max + 0.03), y, f"{lab}  {pace:+,.0f}/yr", color=col,
                fontsize=10.5, fontweight="bold", va="center", ha="left")

    ax.set_xlim(to_date(x_min - 0.06), to_date(x_max + 0.42))
    ax.set_ylim(2300, 4700)
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 7)))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _p: f"{v:,.0f}"))
    ax.tick_params(colors=INK_3, labelsize=10, length=0)
    ax.grid(axis="y", color="#eceef1", linewidth=1)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#d6d8dd")
    ax.set_ylabel("Average score (of 5,000)", color=INK_2, fontsize=10.5)
    ax.set_xlabel("Model release date", color=INK_2, fontsize=10.5)

    ax.set_title(f"GeoBench score over time — {ds['label'].split(' — ')[0]}",
                 loc="left", fontsize=15, fontweight="bold", color=INK, pad=24)
    ax.text(0, 1.015, "Best run per model · trend fitted per lab · /yr = current pace",
            transform=ax.transAxes, fontsize=9.5, color=INK_3, va="bottom")

    fig.tight_layout()
    path = out / f"trends_{ds['id']}.png"
    fig.savefig(path, dpi=180, facecolor="white")
    plt.close(fig)
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=HERE / "timeline_data.json")
    ap.add_argument("--dataset", help="acw or photospheres (default: both)")
    ap.add_argument("--out", type=Path, default=HERE)
    args = ap.parse_args()

    plt.rcParams["font.family"] = "DejaVu Sans"
    data = json.loads(args.data.read_text())
    for key, ds in data["datasets"].items():
        if not args.dataset or key == args.dataset:
            print("wrote", plot(ds, args.out))


if __name__ == "__main__":
    main()
