"""Build the data behind the release-date timeline plot.

Merges three sources into one JSON the timeline page reads:

1. The published leaderboard at https://geobench.org (embedded in its JS bundle).
2. Local ``responses/*/results/summary.json`` from runs that aren't published yet.
3. Model release dates from Epoch AI's ``benchmark_data.zip`` -- the same table
   ``epoch-research/eci-public`` loads (``model_metadata.csv``, plus the
   ``Release date`` column of ``geobench_external.csv``, which is Epoch's own
   mirror of this leaderboard).

Scores are only comparable within a dataset (see ``calculate_score`` in
geobench.py), so results are grouped per dataset and never pooled.

Writes ``timeline_data.json`` and, by folding it into
``timeline.template.html``, the standalone ``timeline.html`` page.

    python -m visualizations.timeline.build_timeline            # from the repo root
    python build_timeline.py --responses ../../responses        # from this directory
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import re
import sys
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
CACHE = HERE / ".cache"

EPOCH_DATA_URL = "https://epoch.ai/data/benchmark_data.zip"
GEOBENCH_SITE = "https://geobench.org"

# geobench.org test id -> (local dataset folder, display label, locations)
DATASETS = {
    "acw-02-25-25": ("acw", "ACW — A Community World", 100),
    "geoguessr-2069-03-27-25": ("photospheres", "Photospheres", 50),
}

# Runs on maps with too few models to make a timeline readable.
SKIP_TESTS = {"avw-03-27-25", "ext-urban-03-26-25", "rural-world-02-27-25"}

# Canonical model identity. Maps every spelling that appears in either source to
# (canonical name, Epoch model_group, variant label).
#
# A "variant" is a run of the same released model at a different reasoning
# effort / thinking budget / tool setting. Variants share a release date and
# collapse to one point -- the plot picks each model's best variant on whichever
# metric is on screen.
MODELS: dict[str, tuple[str, str, str]] = {}


def _m(raw: str, canonical: str, epoch_group: str, variant: str = "") -> None:
    MODELS[raw.lower()] = (canonical, epoch_group, variant)


# -- Anthropic ---------------------------------------------------------------
_m("Claude 3.5 Haiku", "Claude 3.5 Haiku", "Claude 3.5 Haiku")
_m("Claude 3.5 Sonnet", "Claude 3.5 Sonnet", "Claude 3.5 Sonnet (October 2024)")
_m("Claude 3.7 Sonnet", "Claude 3.7 Sonnet", "Claude 3.7 Sonnet")
_m("Claude 3.7 Sonnet (thinking)", "Claude 3.7 Sonnet", "Claude 3.7 Sonnet", "Thinking")
_m("Claude 3.7 Sonnet (Thinking)", "Claude 3.7 Sonnet", "Claude 3.7 Sonnet", "Thinking")
_m("Claude 4 Sonnet", "Claude Sonnet 4", "Claude Sonnet 4")
_m("Claude 4 Sonnet (thinking)", "Claude Sonnet 4", "Claude Sonnet 4", "Thinking")
_m("Claude 4 Opus (thinking)", "Claude Opus 4", "Claude Opus 4", "Thinking")
_m("Claude 4.5 Sonnet (Thinking)", "Claude Sonnet 4.5", "Claude Sonnet 4.5", "Thinking")
_m("Claude 4.5 Opus (Thinking)", "Claude Opus 4.5", "Claude Opus 4.5", "Thinking")
_m("Claude 4.6 Sonnet (Thinking)", "Claude Sonnet 4.6", "Claude Sonnet 4.6", "Thinking")
_m("Claude 4.6 Opus (Thinking)", "Claude Opus 4.6", "Claude Opus 4.6", "Thinking")
_m("Claude 4.8 Opus (Thinking)", "Claude Opus 4.8", "Claude Opus 4.8", "Thinking")
_m("Claude Sonnet 5", "Claude Sonnet 5", "Claude Sonnet 5")
_m("Claude Fable 5", "Claude Fable 5", "Claude Fable 5")
for eff in ("Low", "Medium", "High", "XHigh"):
    _m(f"Claude Opus 5 ({eff})", "Claude Opus 5", "Claude Opus 5", eff)
    _m(f"Claude Fable 5.1 ({eff})", "Claude Fable 5.1", "Claude Fable 5.1", eff)

# -- OpenAI ------------------------------------------------------------------
_m("GPT-4o", "GPT-4o", "GPT-4o (Nov 2024)")
_m("GPT-4o mini", "GPT-4o mini", "GPT-4o mini")
_m("GPT-4.1", "GPT-4.1", "GPT-4.1")
_m("o1", "o1", "o1")
_m("o3", "o3", "o3", "Medium")
_m("o3-high", "o3", "o3", "High")
_m("o4-mini", "o4-mini", "o4-mini", "Medium")
_m("o4-mini-high", "o4-mini", "o4-mini", "High")
_m("GPT-5", "GPT-5", "GPT-5", "Medium")
_m("GPT-5.4 (High)", "GPT-5.4", "GPT-5.4", "High")
for eff in ("Low", "Medium", "High", "XHigh"):
    _m(f"GPT-5 ({eff})", "GPT-5", "GPT-5", eff)
    _m(f"GPT-5.5 ({eff})", "GPT-5.5", "GPT-5.5", eff)
    _m(f"GPT-6 Astra ({eff})", "GPT-6 Astra", "GPT-6 Astra", eff)
    for sub in ("Luna", "Sol", "Terra"):
        _m(f"GPT-5.6 {sub} ({eff})", f"GPT-5.6 {sub}", f"GPT-5.6 {sub}", eff)
for sub in ("Luna", "Sol", "Terra"):
    _m(f"GPT-5.6 {sub}", f"GPT-5.6 {sub}", f"GPT-5.6 {sub}")

# -- Google ------------------------------------------------------------------
_m("Gemini 1.5 Flash", "Gemini 1.5 Flash", "Gemini 1.5 Flash (Sep 2024)")
_m("Gemini 2 Flash", "Gemini 2.0 Flash", "Gemini 2.0 Flash (Feb 2025)")
_m("Gemini 2 Flash Thinking Experimental", "Gemini 2.0 Flash Thinking",
   "Gemini 2.0 Flash Thinking (Jan 2025)")
_m("Gemini 2 Pro Experimental", "Gemini 2.0 Pro", "Gemini 2.0 Pro")
_m("Gemini 2.5 Flash Preview (04-17)", "Gemini 2.5 Flash (Apr 2025)",
   "Gemini 2.5 Flash (Apr 2025)")
_m("Gemini 2.5 Flash Preview (05-20)", "Gemini 2.5 Flash (May 2025)",
   "Gemini 2.5 Flash (May 2025)")
_m("Gemini 2.5 Pro Experimental (03-25)", "Gemini 2.5 Pro (Mar 2025)", "Gemini 2.5 Pro (Mar 2025)")
_m("Gemini 2.5 Pro Experimental (03-25 with Search)", "Gemini 2.5 Pro (Mar 2025)",
   "Gemini 2.5 Pro (Mar 2025)", "Search")
_m("Gemini 2.5 Pro (05-06)", "Gemini 2.5 Pro (May 2025)", "Gemini 2.5 Pro (May 2025)")
_m("Gemini 2.5 Pro (05-06 with Search)", "Gemini 2.5 Pro (May 2025)",
   "Gemini 2.5 Pro (May 2025)", "Search")
_m("Gemini 2.5 Pro", "Gemini 2.5 Pro (Jun 2025)", "Gemini 2.5 Pro (Jun 2025)")
_m("Gemini 3.0 Pro Preview", "Gemini 3 Pro", "Gemini 3 Pro")
_m("Gemini 3.0 Flash Preview", "Gemini 3 Flash", "Gemini 3 Flash")
_m("Gemini 3.1 Pro Preview", "Gemini 3.1 Pro", "Gemini 3.1 Pro")
_m("Gemini 3.1 Flash Lite Preview", "Gemini 3.1 Flash-Lite", "Gemini 3.1 Flash-Lite")
_m("Gemini 3.5 Flash", "Gemini 3.5 Flash", "Gemini 3.5 Flash")
_m("Gemini 3.8 Flash", "Gemini 3.8 Flash", "Gemini 3.8 Flash")
_m("Gemma 3 27b", "Gemma 3 27B", "Gemma 3 27B")
_m("Gemma 3 27B", "Gemma 3 27B", "Gemma 3 27B")

# -- Everyone else -----------------------------------------------------------
_m("Llama 3.2 90b Vision", "Llama 3.2 90B Vision", "Llama 3.2 90B")
_m("Llama 4 Maverick", "Llama 4 Maverick", "Llama 4 Maverick")
_m("Pixtral 12b", "Pixtral 12B", "Pixtral 12B")
_m("Mistral Medium 3.5", "Mistral Medium 3.5", "Mistral Medium 3.5")
_m("Qwen2.5-VL-72B", "Qwen2.5-VL-72B", "Qwen2.5-72B")
_m("Qwen3-VL-235B", "Qwen3-VL-235B", "")
_m("GLM-4.6V", "GLM-4.6V", "")
_m("Grok 4", "Grok 4", "Grok 4")
_m("Grok 4.3", "Grok 4.3", "Grok 4.3 Beta")

# Epoch's own GeoBench sheet names the exact model version it dated. Where it has
# an opinion, take it -- a model_group can span several versions with different
# dates (Gemini 1.5 Flash exp-0827 vs -002), and the group date is then ambiguous.
EPOCH_VERSIONS = {
    "Claude 3.5 Haiku": "claude-3-5-haiku-20241022",
    "Claude 3.5 Sonnet": "claude-3-5-sonnet-20241022",
    "Claude 3.7 Sonnet": "claude-3-7-sonnet-20250219",
    "Claude Sonnet 4": "claude-sonnet-4-20250514",
    "Claude Opus 4": "claude-opus-4-20250514",
    "Claude Opus 4.5": "claude-opus-4-5-20251101",
    "GPT-4o": "gpt-4o-2024-11-20",
    "GPT-4o mini": "gpt-4o-mini-2024-07-18",
    "GPT-4.1": "gpt-4.1-2025-04-14",
    "o1": "o1-2024-12-17",
    "o3": "o3-2025-04-16",
    "o4-mini": "o4-mini-2025-04-16",
    "GPT-5": "gpt-5-2025-08-07",
    "Gemini 1.5 Flash": "gemini-1.5-flash-002",
    "Gemini 2.0 Flash": "gemini-2.0-flash-001",
    "Gemini 2.5 Flash (Apr 2025)": "gemini-2.5-flash-preview-04-17",
    "Gemini 2.5 Flash (May 2025)": "gemini-2.5-flash-preview-05-20",
    "Gemini 2.5 Pro (Mar 2025)": "gemini-2.5-pro-exp-03-25",
    "Gemini 2.5 Pro (May 2025)": "gemini-2.5-pro-preview-05-06",
    "Gemini 3 Pro": "gemini-3-pro-preview",
    "Gemini 3 Flash": "gemini-3-flash-preview",
    "Gemma 3 27B": "gemma-3-27b-it",
    "Llama 3.2 90B Vision": "Llama-3.2-90B-Vision-Instruct",
    "Llama 4 Maverick": "Llama-4-Maverick-17B-128E-Instruct",
    "Pixtral 12B": "Pixtral-12B-2409",
    "Grok 4": "grok-4-0709",
}

# Release dates for models Epoch's tables don't carry. Sourced by hand; the plot
# marks these so they're never mistaken for Epoch's numbers.
MANUAL_DATES = {
    "Qwen3-VL-235B": ("2025-09-23", "Qwen release announcement"),
    "GLM-4.6V": ("2025-12-09", "Z.ai release announcement"),
}

ORGS = {
    "Anthropic": "Anthropic",
    "OpenAI": "OpenAI",
    "Google": "Google",
    "Google DeepMind": "Google",
    "Meta": "Meta",
    "Alibaba": "Alibaba",
    "Mistral": "Mistral",
    "xAI": "xAI",
    "Z.ai": "Z.ai",
}

# provider isn't on every published row; fall back to the model name.
NAME_ORG = [
    ("claude", "Anthropic"),
    ("gpt-", "OpenAI"), ("o1", "OpenAI"), ("o3", "OpenAI"), ("o4-", "OpenAI"),
    ("gemini", "Google"), ("gemma", "Google"),
    ("llama", "Meta"),
    ("qwen", "Alibaba"),
    ("pixtral", "Mistral"), ("mistral", "Mistral"),
    ("grok", "xAI"),
    ("glm", "Z.ai"),
]


def fetch(url: str, cache_name: str, binary: bool = False, refresh: bool = False) -> bytes:
    CACHE.mkdir(exist_ok=True)
    path = CACHE / cache_name
    if path.exists() and not refresh:
        return path.read_bytes()
    req = Request(url, headers={"User-Agent": "geobench-timeline/1.0"})
    with urlopen(req) as resp:
        data = resp.read()
    path.write_bytes(data)
    return data


def load_epoch_dates(refresh: bool = False) -> dict[str, str]:
    """Release dates from Epoch's benchmark_data.zip, by group and by version."""
    blob = fetch(EPOCH_DATA_URL, "benchmark_data.zip", binary=True, refresh=refresh)
    seen: dict[str, Counter] = defaultdict(Counter)
    version_dates: dict[str, str] = {}
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        with z.open("model_metadata.csv") as f:
            for row in csv.DictReader(io.TextIOWrapper(f, "utf-8")):
                group, date = row.get("model_group", ""), row.get("date", "")
                if group and date:
                    seen[group][date[:10]] += 1
                version = row.get("model_version", "")
                if version and date:
                    version_dates.setdefault(version, date[:10])
        # Epoch's GeoBench sheet is the most on-point source for these models,
        # so its Release date wins over the generic metadata table.
        with z.open("geobench_external.csv") as f:
            for row in csv.DictReader(io.TextIOWrapper(f, "utf-8")):
                version, date = row.get("Model version", ""), row.get("Release date", "")
                if version and date:
                    version_dates[version] = date[:10]
    # A handful of groups carry more than one date across their rows; take the
    # modal date, earliest on a tie, so the result is stable run to run.
    group_dates = {g: min(c.most_common(), key=lambda kv: (-kv[1], kv[0]))[0]
                   for g, c in seen.items()}
    return group_dates, version_dates


def load_published(refresh: bool = False) -> list[dict]:
    """Scrape the leaderboard rows embedded in geobench.org's JS bundle."""
    html = fetch(GEOBENCH_SITE, "geobench_index.html", refresh=refresh).decode("utf-8", "replace")
    m = re.search(r'src="(/assets/index-[^"]+\.js)"', html)
    if not m:
        raise SystemExit("could not find the JS bundle on geobench.org")
    js = fetch(GEOBENCH_SITE + m.group(1), "geobench_bundle.js", refresh=refresh)
    js = js.decode("utf-8", "replace")

    for hit in re.finditer(r"JSON\.parse\('", js):
        start = hit.end()
        i = start
        while i < len(js):
            if js[i] == "\\":
                i += 2
                continue
            if js[i] == "'":
                break
            i += 1
        raw = js[start:i]
        try:
            decoded = raw.encode().decode("unicode_escape").encode("latin1").decode("utf-8")
            data = json.loads(decoded)
        except Exception:
            continue
        if isinstance(data, list) and data and "country_success_rate" in data[0]:
            return data
    raise SystemExit("could not find the leaderboard JSON in the bundle")


def load_local(responses_dir: Path) -> list[dict]:
    rows = []
    if not responses_dir.is_dir():
        print(f"  ! no responses dir at {responses_dir}", file=sys.stderr)
        return rows
    for summary in sorted(responses_dir.glob("*/results/summary.json")):
        try:
            rows.append(json.loads(summary.read_text()))
        except Exception as exc:
            print(f"  ! {summary}: {exc}", file=sys.stderr)
    return rows


def org_for(name: str, provider: str | None) -> str:
    if provider and provider in ORGS:
        return ORGS[provider]
    low = name.lower()
    for prefix, org in NAME_ORG:
        if low.startswith(prefix):
            return org
    return "Other"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--responses", type=Path, default=REPO_ROOT / "responses",
                    help="directory of benchmark run folders")
    ap.add_argument("--out", type=Path, default=HERE / "timeline_data.json")
    ap.add_argument("--refresh", action="store_true", help="re-download cached sources")
    args = ap.parse_args()

    print("Loading Epoch release dates ...")
    epoch_groups, epoch_versions = load_epoch_dates(args.refresh)
    print(f"  {len(epoch_groups)} model groups, {len(epoch_versions)} model versions")

    print("Loading published geobench.org results ...")
    published = load_published(args.refresh)
    print(f"  {len(published)} rows")

    print(f"Loading local runs from {args.responses} ...")
    local = load_local(args.responses)
    print(f"  {len(local)} rows")

    # test id -> canonical model -> accumulated variants
    buckets: dict[str, dict[str, dict]] = {k: {} for k in DATASETS}
    unmapped: set[str] = set()
    undated: set[str] = set()

    def ingest(row: dict, source: str) -> None:
        test = row.get("test", "")
        # local runs name the dataset folder; published rows name the test id
        if test in DATASETS:
            test_id = test
        else:
            match = [t for t, (folder, _, _) in DATASETS.items() if folder == test]
            if not match:
                if test not in SKIP_TESTS and not test.endswith("_region"):
                    unmapped.add(f"test:{test}")
                return
            test_id = match[0]

        raw = row.get("model", "")
        entry = MODELS.get(raw.lower())
        if entry is None:
            unmapped.add(f"model:{raw}")
            return
        canonical, epoch_group, variant = entry

        version = EPOCH_VERSIONS.get(canonical)
        date = epoch_versions.get(version) if version else None
        if date is None and epoch_group:
            date = epoch_groups.get(epoch_group)
        date_source = "Epoch AI"
        if date is None and canonical in MANUAL_DATES:
            date, date_source = MANUAL_DATES[canonical]
        if date is None:
            undated.add(f"{canonical} (epoch group: {epoch_group or '-'})")
            return

        bucket = buckets[test_id].setdefault(canonical, {
            "model": canonical,
            "org": org_for(canonical, row.get("provider")),
            "date": date,
            "date_source": date_source,
            "epoch_group": epoch_group,
            "variants": [],
        })
        # A non-standard prompt_mode is extra help, like search -- label it so it
        # cannot be mistaken for another run of the same setup, and let the page
        # filter it out.
        mode = (row.get("prompt_mode") or "standard").strip()
        tuned = mode not in ("", "standard")
        label = variant or "default"
        if tuned:
            label = f"{variant} · {mode} prompt" if variant else f"{mode} prompt"

        bucket["variants"].append({
            "variant": label,
            "score": round(float(row["average_score"]), 1),
            "country": round(float(row["country_success_rate"]), 4),
            "median_score": row.get("median_score"),
            "median_distance_km": (round(float(row["median_distance_km"]), 1)
                                   if row.get("median_distance_km") is not None else None),
            "refusal": round(float(row.get("refusal_rate", 0)), 4),
            "n": int(row.get("n", 0)),
            "tools": variant == "Search",
            "prompt_tuned": tuned,
            "source": source,
        })

    for row in published:
        ingest(row, "geobench.org")
    for row in local:
        ingest(row, "local run")

    datasets = {}
    for test_id, (folder, label, n_locs) in DATASETS.items():
        models = sorted(buckets[test_id].values(), key=lambda m: (m["date"], m["model"]))
        for m in models:
            # stable order so the tooltip reads consistently
            m["variants"].sort(key=lambda v: -v["score"])
        if not models:
            continue
        datasets[folder] = {
            "id": folder,
            "test_id": test_id,
            "label": label,
            "locations": n_locs,
            "models": models,
        }
        print(f"  {label}: {len(models)} models, "
              f"{sum(len(m['variants']) for m in models)} runs")

    out = {
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "sources": {
            "scores": "geobench.org leaderboard + local benchmark runs",
            "dates": "Epoch AI benchmark_data.zip (model_metadata.csv)",
            "dates_url": EPOCH_DATA_URL,
        },
        "datasets": datasets,
    }
    args.out.write_text(json.dumps(out, indent=1))
    print(f"\nWrote {args.out}")

    template = HERE / "timeline.template.html"
    if template.exists():
        page = template.read_text().replace(
            "__DATA__", json.dumps(out, separators=(",", ":")))
        page_path = HERE / "timeline.html"
        page_path.write_text(page)
        print(f"Wrote {page_path}  ({len(page) / 1024:.0f} KB)")

    if unmapped:
        print("\nUnmapped (add to MODELS/DATASETS if these should appear):")
        for u in sorted(unmapped):
            print(f"  {u}")
    if undated:
        print("\nNo release date found (add to MANUAL_DATES):")
        for u in sorted(undated):
            print(f"  {u}")


if __name__ == "__main__":
    main()
