"""Static plot: GeoBench score against release date, with a trend per lab.

Reads timeline_data.json (run build_timeline.py first) and writes
``trends_<dataset>.png`` next to this file.

Each point is one OpenAI, Google or Anthropic model at its best run; search and
tuned-prompt runs are left out so every lab is measured on the same setup. The
trend line is fitted in logit space against the 5,000-point ceiling -- the
sigmoid shape Epoch's ECI model assumes -- because a straight line on a bounded
score eventually predicts the impossible. Each line spans only the dates that
lab has actually shipped in. Every model is drawn, but each line is fitted only through its lab's best-so-far
models -- one that didn't beat the lab's previous best (a small or cheap variant)
is drawn in a lighter tint and kept out of the fit. The first and latest best-so-far are named.

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
from matplotlib import patheffects

HERE = Path(__file__).resolve().parent
CEILING = 5000.0

LABS = {"OpenAI": "#2a78d6", "Google": "#eb6834", "Anthropic": "#1baf7a"}
INK, INK_2, INK_3 = "#16181c", "#4d525c", "#80848c"

# "Professional player" average score, from the human baselines the
# geobench.org leaderboard publishes alongside the model results.
PRO_HUMAN = {"acw": 4100, "photospheres": 3700}

YEAR = 365.2425


def tint(hex_colour: str, amount: float) -> tuple[float, float, float]:
    """The colour mixed toward white: amount=1 is the colour itself, 0 is white."""
    r, g, b = (int(hex_colour[i:i + 2], 16) / 255 for i in (1, 3, 5))
    return tuple(1 - amount * (1 - c) for c in (r, g, b))


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


def best_scores(ds: dict, lab: str) -> list[tuple[float, float, str]]:
    """(release year, best standard-setup score, model name) for one lab."""
    out = []
    for m in ds["models"]:
        if m["org"] != lab:
            continue
        runs = [v for v in m["variants"] if not v.get("tools") and not v.get("prompt_tuned")]
        if runs:
            out.append((to_year(m["date"]), max(v["score"] for v in runs), m["model"]))
    return sorted(out)


def lab_frontier(pts: list[tuple[float, float, str]]) -> list[tuple[float, float, str]]:
    """The lab's best-so-far models: each one that beat everything the same lab
    had released before it. A lab's small, cheap models never make this list,
    so they don't drag its trend down."""
    out, best = [], -math.inf
    for x, y, name in sorted(pts, key=lambda p: (p[0], -p[1])):
        if y > best:
            out.append((x, y, name))
            best = y
    return out


def notable(frontiers: dict[str, list[tuple[float, float, str]]]) -> set[str]:
    """Models worth naming: the first and latest model on each lab's frontier."""
    names = set()
    for fr in frontiers.values():
        names.add(fr[0][2])
        names.add(fr[-1][2])
    return names


def place_labels(ax, fig, items, avoid, obstacles) -> None:
    """Greedy label placement: try spots around each point, nearest first, and
    keep the first that overlaps no earlier label, nothing in `avoid`, and none
    of the `obstacles` -- display-space points sampled along the lines."""
    renderer = fig.canvas.get_renderer()
    taken = [b.expanded(1.04, 1.15) for b in avoid]
    offsets = [(0, 11, "center", "bottom"), (0, -11, "center", "top"),
               (9, 0, "left", "center"), (-9, 0, "right", "center"),
               (0, 24, "center", "bottom"), (0, -24, "center", "top"),
               (9, 14, "left", "bottom"), (-9, 14, "right", "bottom"),
               (9, -14, "left", "top"), (-9, -14, "right", "top")]
    for x, y, name, col in sorted(items, key=lambda t: -t[1]):
        for dx, dy, ha, va in offsets:
            t = ax.annotate(name, (x, y), xytext=(dx, dy), textcoords="offset points",
                            ha=ha, va=va, fontsize=8.5, color=col, fontweight="bold",
                            path_effects=[patheffects.withStroke(linewidth=3, foreground="white")],
                            zorder=6)
            bb = t.get_window_extent(renderer)
            inside = ax.get_window_extent(renderer).contains(bb.x0, bb.y0) and \
                ax.get_window_extent(renderer).contains(bb.x1, bb.y1)
            hits_line = any(bb.x0 - 3 <= px <= bb.x1 + 3 and bb.y0 - 3 <= py <= bb.y1 + 3
                            for px, py in obstacles)
            if inside and not hits_line and not any(bb.overlaps(o) for o in taken):
                taken.append(bb.expanded(1.04, 1.15))
                break
            t.remove()


def plot(ds: dict, out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(11, 6.4))
    series = {lab: best_scores(ds, lab) for lab in LABS}
    series = {lab: pts for lab, pts in series.items() if len(lab_frontier(pts)) >= 2}
    all_x = [p[0] for pts in series.values() for p in pts]
    all_y = [p[1] for pts in series.values() for p in pts]
    x_min, x_max = min(all_x), max(all_x)
    ends = []
    lines = []      # (dates, values) of everything a label must not sit on

    frontiers = {lab: lab_frontier(pts) for lab, pts in series.items()}
    for lab, pts in series.items():
        col = LABS[lab]
        fr = frontiers[lab]
        on_fr = {p[2] for p in fr}

        # the rest of the lab's line-up stays on the plot, but not in the line
        rest = [p for p in pts if p[2] not in on_fr]
        ax.scatter([to_date(p[0]) for p in rest], [p[1] for p in rest], s=26,
                   color=tint(col, 0.45), edgecolor=tint(col, 0.75), linewidth=1, zorder=3)

        xs, ys, _ = zip(*fr)
        a, b = logistic_fit(xs, ys)
        grid = [xs[0] + (xs[-1] - xs[0]) * i / 100 for i in range(101)]
        curve = [CEILING / (1 + math.exp(-(a + b * g))) for g in grid]

        ax.scatter([to_date(x) for x in xs], ys, s=62, color=col,
                   edgecolor="white", linewidth=1.2, zorder=5)
        ax.plot([to_date(g) for g in grid], curve, color=col, linewidth=2.8, zorder=4)
        ends.append([curve[-1], lab, col, grid[-1]])
        lines.append(([to_date(g) for g in grid], curve))

    # pro human baseline
    human = PRO_HUMAN.get(ds["id"])
    avoid = []
    if human:
        # stop short of the lab names so the line never runs through them
        ax.hlines(human, to_date(x_min - 0.06), to_date(x_max + 0.01), color=INK_2,
                  linewidth=1.4, linestyle=(0, (5, 4)), zorder=2)
        hx = [x_min - 0.06 + (x_max - x_min + 0.07) * i / 200 for i in range(201)]
        lines.append(([to_date(v) for v in hx], [human] * len(hx)))
        avoid.append(ax.text(to_date(x_min), human + 25, f"Pro human  {human:,}",
                             color=INK_2, fontsize=10, ha="left", va="bottom"))

    # each lab's name sits at the end of its own line -- a lab whose best model is
    # older (no later model beat it) ends earlier. Names that would land on top
    # of each other are spread apart vertically.
    ends.sort()
    for i in range(1, len(ends)):
        for j in range(i):
            if abs(ends[i][3] - ends[j][3]) < 0.3:
                ends[i][0] = max(ends[i][0], ends[j][0] + 110)
    for y, lab, col, x_end in ends:
        avoid.append(ax.text(to_date(x_end + 0.03), y, lab, color=col, fontsize=11,
                             fontweight="bold", va="center", ha="left"))

    ax.set_xlim(to_date(x_min - 0.06), to_date(x_max + 0.32))
    ax.set_ylim(math.floor((min(all_y) - 150) / 250) * 250,
                min(CEILING, math.ceil((max(all_y) + 200) / 250) * 250))
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
    ax.text(0, 1.015, "Solid dots: each lab's best model at the time, which the line is "
            "fitted through · light dots: its other models",
            transform=ax.transAxes, fontsize=9.5, color=INK_3, va="bottom")

    fig.tight_layout()
    fig.canvas.draw()
    named = notable(frontiers)
    items = [(to_date(x), y, name, LABS[lab]) for lab, fr in frontiers.items()
             for x, y, name in fr if name in named]
    obstacles = []
    for dates, vals in lines:
        for d, v in zip(dates, vals):
            obstacles.append(tuple(ax.transData.transform((mdates.date2num(d), v))))
    # and every marker, so no label hides another model's point
    for pts in series.values():
        for x, y, _ in pts:
            obstacles.append(tuple(ax.transData.transform((mdates.date2num(to_date(x)), y))))
    place_labels(ax, fig, items,
                 [t.get_window_extent(fig.canvas.get_renderer()) for t in avoid], obstacles)

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
