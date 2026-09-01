"""Build a self-contained HTML viewer for model guesses.

Reads `responses/<Model>_<dataset>_<timestamp>/` run folders plus the
`dataset/<dataset>/metadata.json` they were scored against, and writes a single
HTML file pairing every location's image with the truth, each model's guess and
the model's own reasoning.

    python visualizations/viewer/build.py                 # every dataset with runs
    python visualizations/viewer/build.py -d poland       # just one
    python visualizations/viewer/build.py -d poland japan --serve

With more than one dataset the page grows a dataset switcher and a cross-dataset
comparison; everything else works the same.

The output references dataset images by relative path, so it opens straight from
disk (file://) with no server. Use --serve if your browser blocks that.
"""

import argparse
import csv
import functools
import http.server
import json
import math
import os
import re
import socketserver
import sys
import webbrowser
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
TEMPLATE = HERE / "template.html"

# Same derivation geobench.py uses, so the bar in the UI matches the CSV.
SCORE_BASE = 0.99866017
SCALE_DIVISOR = 7.458421


def haversine_km(lat1, lng1, lat2, lng2):
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lng2 - lng1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def to_float(value):
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) else f


def to_bool(value):
    return str(value).strip().lower() in ("true", "1", "yes")


def read_json(path, default=None):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return default


def load_dataset(dataset_dir):
    meta = read_json(dataset_dir / "metadata.json")
    if meta is None:
        raise SystemExit(f"no readable metadata.json in {dataset_dir}")

    bounds = meta.get("bounds") or {}
    locations = {}
    for item in meta.get("images", []):
        rel = item.get("image_path", "")
        loc_id = Path(rel).stem
        locations[loc_id] = {
            "id": loc_id,
            "image": (dataset_dir / rel).resolve(),
            "country": item.get("country"),
            "lat": item.get("lat"),
            "lng": item.get("lng"),
            "pano": item.get("pano_id"),
            "heading": item.get("heading"),
            "date": item.get("image_date"),
        }
    return bounds, locations


def scale_from_bounds(bounds):
    """Scoring scale in km: the map diagonal over the GeoGuessr constant."""
    try:
        diagonal = haversine_km(
            bounds["min"]["lat"], bounds["min"]["lng"],
            bounds["max"]["lat"], bounds["max"]["lng"],
        )
    except (KeyError, TypeError):
        return None
    return diagonal / SCALE_DIVISOR


# --- per-run extraction -----------------------------------------------------

THINK_RE = re.compile(r"<thinking>(.*?)</thinking>", re.DOTALL | re.IGNORECASE)
ANSWER_RE = re.compile(
    r"^[ \t]*\**[ \t]*(country|lat|lng)[ \t]*\**[ \t]*[:=].*$",
    re.IGNORECASE | re.MULTILINE,
)


def split_response(text):
    """Separate a response into thinking, prose and the trailing answer block."""
    thinking = "\n\n".join(m.strip() for m in THINK_RE.findall(text))
    body = THINK_RE.sub("", text).strip()

    matches = list(ANSWER_RE.finditer(body))
    if not matches:
        return thinking, body, ""

    # Walk back from the last answer line while the lines stay contiguous, so a
    # model that mentions "lat:" mid-reasoning does not swallow the whole text.
    block = [matches[-1]]
    for prev in reversed(matches[:-1]):
        gap = body[prev.end():block[0].start()]
        if gap.strip():
            break
        block.insert(0, prev)

    prose = body[:block[0].start()].strip()
    answer = body[block[0].start():].strip()
    return thinking, prose, answer


def extract_usage(raw):
    """Best-effort token / latency stats across the provider response shapes."""
    if not isinstance(raw, dict):
        return {}
    out = {}

    usage = raw.get("usage") or raw.get("usageMetadata") or {}
    if isinstance(usage, dict):
        out["inTok"] = (usage.get("input_tokens") or usage.get("prompt_tokens")
                        or usage.get("promptTokenCount"))
        out["outTok"] = (usage.get("output_tokens") or usage.get("completion_tokens")
                         or usage.get("candidatesTokenCount"))
        details = (usage.get("output_tokens_details")
                   or usage.get("completion_tokens_details") or {})
        if isinstance(details, dict) and details.get("reasoning_tokens"):
            out["reasonTok"] = details["reasoning_tokens"]
        if usage.get("thoughtsTokenCount"):
            out["reasonTok"] = usage["thoughtsTokenCount"]

    created = raw.get("created_at") or raw.get("created")
    done = raw.get("completed_at")
    if isinstance(created, (int, float)) and isinstance(done, (int, float)):
        out["latency"] = round(done - created, 1)

    return {k: v for k, v in out.items() if v}


def load_run(run_dir, locations, scale):
    csv_path = run_dir / "results" / "detailed.csv"
    if not csv_path.exists():
        return None
    summary = read_json(run_dir / "results" / "summary.json", {}) or {}

    guesses = {}
    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            loc_id = str(row.get("location_id", "")).strip()
            if loc_id not in locations:
                continue

            refused = to_bool(row.get("refused"))
            lat, lng = to_float(row.get("lat_guess")), to_float(row.get("lng_guess"))
            dist = to_float(row.get("distance_km"))
            if dist is None and not refused and None not in (lat, lng):
                truth = locations[loc_id]
                dist = haversine_km(truth["lat"], truth["lng"], lat, lng)

            score = to_float(row.get("score"))
            if score is None and dist is not None and scale:
                score = round(5000 * SCORE_BASE ** (dist / scale))

            text = ""
            out_file = run_dir / "output" / f"{loc_id}.txt"
            if out_file.exists():
                text = out_file.read_text(encoding="utf-8", errors="replace")
            thinking, prose, answer = split_response(text)

            entry = {
                "id": loc_id,
                "refused": refused,
                "error": (row.get("error_message") or "").strip() or None,
                "country": (row.get("country_guess") or "").strip() or None,
                "countryOk": to_bool(row.get("country_correct")),
                "lat": lat,
                "lng": lng,
                "km": round(dist, 2) if dist is not None else None,
                "score": int(score) if score is not None else 0,
                "thinking": thinking or None,
                "text": prose,
                "answer": answer or None,
            }
            entry.update(extract_usage(read_json(run_dir / "json" / f"{loc_id}.json")))
            guesses[loc_id] = entry

    if not guesses:
        return None

    stamp = run_dir.name.rsplit("_", 1)[-1]
    return {
        "id": run_dir.name,
        "name": summary.get("model") or run_dir.name.split("_")[0],
        "provider": summary.get("provider") or "Unknown",
        "promptMode": summary.get("prompt_mode") or "standard",
        "run": stamp.replace("T", " ").replace("_", ":"),
        "guesses": guesses,
    }


def discover_runs(responses_dir, dataset, only=None):
    if not responses_dir.is_dir():
        raise SystemExit(f"no responses directory at {responses_dir}")

    runs = []
    for path in sorted(responses_dir.iterdir()):
        if not path.is_dir():
            continue
        # <model>_<dataset>[_region]_<timestamp>
        if dataset not in path.name.split("_")[1:]:
            continue
        if only and not any(o.lower() in path.name.lower() for o in only):
            continue
        runs.append(path)
    return runs


# --- render -----------------------------------------------------------------

def datasets_with_runs(dataset_root, responses_dir):
    """Every dataset directory that at least one run folder refers to."""
    if not responses_dir.is_dir():
        raise SystemExit(f"no responses directory at {responses_dir}")
    named = set()
    for path in responses_dir.iterdir():
        if path.is_dir():
            named.update(path.name.split("_")[1:])
    return sorted(d.name for d in dataset_root.iterdir()
                  if d.is_dir() and d.name in named)


def build_dataset(dataset, dataset_dir, responses_dir, out_dir, only=None):
    bounds, locations = load_dataset(dataset_dir)
    scale = scale_from_bounds(bounds)

    run_dirs = discover_runs(responses_dir, dataset, only)
    if not run_dirs:
        raise SystemExit(f"no runs for dataset '{dataset}' in {responses_dir}")

    models = []
    for run_dir in run_dirs:
        run = load_run(run_dir, locations, scale)
        if run:
            models.append(run)
        else:
            print(f"  skipped {run_dir.name} (no usable results)", file=sys.stderr)
    if not models:
        raise SystemExit(f"no usable runs for dataset '{dataset}'")

    # Keep only locations at least one model actually answered on.
    answered = {k for m in models for k in m["guesses"]}
    ordered = sorted((l for k, l in locations.items() if k in answered),
                     key=lambda l: (len(l["id"]), l["id"]))

    for loc in ordered:
        rel = os.path.relpath(loc.pop("image"), out_dir)
        loc["src"] = rel.replace(os.sep, "/")

    models.sort(key=lambda m: (m["name"].lower(), m["run"]))
    return {
        "name": dataset,
        "bounds": bounds,
        "scale": round(scale, 3) if scale else None,
        "locations": ordered,
        "models": models,
    }


def render(payload, out_path):
    template = TEMPLATE.read_text(encoding="utf-8")
    if "/*__DATA__*/" not in template:
        raise SystemExit(f"{TEMPLATE} is missing the /*__DATA__*/ placeholder")
    blob = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    # Keep the payload from terminating the surrounding <script>.
    blob = blob.replace("</", "<\\/")
    out_path.write_text(template.replace("/*__DATA__*/", blob), encoding="utf-8")
    return out_path


def serve(out_path, dataset_root, port):
    # The page points at dataset images *above* its own directory, so the server
    # has to be rooted where both live or every image 404s.
    root = Path(os.path.commonpath([out_path.parent, dataset_root]))
    rel = out_path.relative_to(root).as_posix()
    handler = functools.partial(http.server.SimpleHTTPRequestHandler,
                                directory=str(root))
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("127.0.0.1", port), handler) as httpd:
        url = f"http://127.0.0.1:{port}/{rel}"
        print(f"serving {root}\n  open {url}\n  (ctrl-c to stop)", flush=True)
        webbrowser.open(url)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print()


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("-d", "--dataset", nargs="*", default=None, metavar="NAME",
                    help="dataset name(s), e.g. poland japan "
                         "(default: every dataset that has runs)")
    ap.add_argument("-m", "--models", nargs="*", default=None,
                    help="only include runs whose folder name contains these strings")
    ap.add_argument("--responses", type=Path, default=None,
                    help="responses directory (default: <repo>/responses)")
    ap.add_argument("--dataset-root", type=Path, default=None,
                    help="dataset directory root (default: <repo>/dataset)")
    ap.add_argument("-o", "--out", type=Path, default=None,
                    help="output path (default: out/<dataset>.html, "
                         "or out/geobench.html for several)")
    ap.add_argument("--serve", action="store_true",
                    help="serve the output over http and open a browser")
    ap.add_argument("--port", type=int, default=8020)
    ap.add_argument("--open", action="store_true", help="open the file directly")
    args = ap.parse_args()

    responses_dir = (args.responses or REPO_ROOT / "responses").resolve()
    dataset_root = (args.dataset_root or REPO_ROOT / "dataset").resolve()
    if not dataset_root.is_dir():
        raise SystemExit(f"no dataset directory at {dataset_root}")

    names = args.dataset or datasets_with_runs(dataset_root, responses_dir)
    if not names:
        raise SystemExit(f"no dataset under {dataset_root} has runs in {responses_dir}")
    for name in names:
        if not (dataset_root / name).is_dir():
            raise SystemExit(f"no dataset directory at {dataset_root / name}")

    default_name = f"{names[0]}.html" if len(names) == 1 else "geobench.html"
    out_path = (args.out or HERE / "out" / default_name).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    datasets = [build_dataset(name, dataset_root / name, responses_dir,
                              out_path.parent, args.models)
                for name in names]
    render({"datasets": datasets}, out_path)

    print(f"{out_path}  ({out_path.stat().st_size / 1e6:.1f} MB)")
    for ds in datasets:
        print(f"  {ds['name']:<12} {len(ds['locations']):>3} locations x "
              f"{len(ds['models'])} models: "
              + ", ".join(m["name"] for m in ds["models"]))

    if args.serve:
        serve(out_path, dataset_root, args.port)
    elif args.open:
        webbrowser.open(out_path.as_uri())


if __name__ == "__main__":
    main()
