"""Play the benchmark yourself: a few of the dataset's locations, your own pin,
your own reasoning, scored exactly the way the models are.

    python human/quiz.py                        # 10 random acw locations
    python human/quiz.py -d poland -n 5         # five from another dataset
    python human/quiz.py --ids 3 7 12           # pick the locations yourself
    python human/quiz.py --resume responses/julian_acw_2026-09-14T10_00_00

It serves a page on 127.0.0.1 and opens a browser: the image on one side, a map
and a reasoning box on the other. Submit a round and it reveals the true
location, your distance and score, and what every model in `responses/` said
about the same image. At the end you are ranked against those models over the
same locations.

The session writes an ordinary run folder - `responses/<name>_<dataset>_<stamp>/`
with `output/<id>.txt`, `results/detailed.csv` and `results/summary.json` - after
every round, so the viewer, `scripts/fix.py` and the dashboards treat you as just
another model, and a session you abandon halfway still counts.

The page is never told the answers: the truth, the scoring and the model
comparison stay on the server and arrive only once a round has been submitted.
"""

import argparse
import csv
import datetime
import http.server
import json
import math
import mimetypes
import os
import random
import re
import sys
import threading
import webbrowser
from pathlib import Path
from urllib.parse import urlparse

import haversine
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
TEMPLATE = HERE / "template.html"

sys.path.insert(0, str(REPO_ROOT))

from geobench import BenchmarkResult, Location  # noqa: E402
from geo2p.canon import COUNTRY_GROUPS  # noqa: E402
from scripts.parser import Guess  # noqa: E402

# geobench.py derives the GeoGuessr scale from the map diagonal the same way;
# the scoring itself comes from geobench.calculate_score via BenchmarkResult, so
# the only thing repeated here is the divisor.
SCALE_DIVISOR = 7.458421

THINK_RE = re.compile(r"<thinking>(.*?)</thinking>", re.DOTALL | re.IGNORECASE)


def read_json(path, default=None):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return default


def to_float(value):
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) else f


def to_bool(value):
    return str(value).strip().lower() in ("true", "1", "yes")


def load_dataset(dataset_dir):
    """Locations and the scoring scale, exactly as geobench.py reads them."""
    meta = read_json(dataset_dir / "metadata.json")
    if meta is None:
        raise SystemExit(f"no readable metadata.json in {dataset_dir}")

    bounds = meta["bounds"]
    scale = haversine.haversine(
        (bounds["min"]["lat"], bounds["min"]["lng"]),
        (bounds["max"]["lat"], bounds["max"]["lng"]),
    ) / SCALE_DIVISOR

    locations = [
        Location(
            image_path=str(dataset_dir / item["image_path"]),
            country=item["country"],
            lat=item["lat"],
            lng=item["lng"],
        )
        for item in meta["images"]
    ]
    return bounds, scale, locations


# --- what the models said ---------------------------------------------------

def discover_runs(responses_dir, dataset, only=None, exclude=None):
    """Run folders belonging to a dataset: <model>_<dataset>[_mode]_<stamp>."""
    if not responses_dir.is_dir():
        return []
    out = []
    for path in sorted(responses_dir.iterdir()):
        if not path.is_dir() or path.name == exclude:
            continue
        if dataset not in path.name.split("_")[1:]:
            continue
        if only and not any(o.lower() in path.name.lower() for o in only):
            continue
        out.append(path)
    return out


def load_run(run_dir, wanted, truth, scale):
    """One run's guesses on the quiz locations, or None if it has none."""
    csv_path = run_dir / "results" / "detailed.csv"
    if not csv_path.exists():
        return None

    guesses = {}
    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            loc_id = str(row.get("location_id", "")).strip()
            if loc_id not in wanted:
                continue

            refused = to_bool(row.get("refused"))
            lat, lng = to_float(row.get("lat_guess")), to_float(row.get("lng_guess"))
            if not refused and None in (lat, lng):
                refused = True

            km = to_float(row.get("distance_km"))
            score = to_float(row.get("score"))
            if not refused and (km is None or score is None):
                # A hand-edited CSV can be missing them; rescore the same way
                # geobench.py would rather than drop the run.
                rescored = BenchmarkResult(location=truth[loc_id],
                                           guess=Guess("", lat, lng))
                rescored.calculate_metrics(scale)
                km = rescored.distance_km if km is None else km
                score = rescored.score if score is None else score

            guesses[loc_id] = {
                "refused": refused,
                "country": (row.get("country_guess") or "").strip() or None,
                "countryOk": to_bool(row.get("country_correct")),
                "lat": lat,
                "lng": lng,
                "km": round(km, 1) if km is not None else None,
                "score": int(score) if score is not None else 0,
                "error": (row.get("error_message") or "").strip() or None,
            }

    if not guesses:
        return None

    summary = read_json(run_dir / "results" / "summary.json", {}) or {}
    mode = summary.get("prompt_mode") or "standard"
    name = summary.get("model") or run_dir.name.split("_")[0]
    return {
        "dir": run_dir,
        "name": name if mode == "standard" else f"{name} [{mode}]",
        "provider": summary.get("provider") or "Unknown",
        "guesses": guesses,
    }


def reasoning_for(run, loc_id):
    """A model's own words for one location, minus any thinking block."""
    path = run["dir"] / "output" / f"{loc_id}.txt"
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    return THINK_RE.sub("", text).strip()


# --- the session ------------------------------------------------------------

class Quiz:
    def __init__(self, dataset, dataset_dir, responses_dir, rounds, name,
                 run_dir, seed, only=None):
        self.dataset = dataset
        self.responses_dir = responses_dir
        self.name = name
        self.run_dir = run_dir
        self.seed = seed
        self.rounds = rounds
        self.lock = threading.Lock()
        self.answers = {}

        self.bounds, self.scale, all_locations = load_dataset(dataset_dir)
        self.truth = {loc.id: loc for loc in self.rounds}
        self.runs = [
            run for run in (
                load_run(d, set(self.truth), self.truth, self.scale)
                for d in discover_runs(responses_dir, dataset, only,
                                       exclude=run_dir.name)
            ) if run
        ]
        self.total_locations = len(all_locations)

    # -- playing --

    def submit(self, index, lat, lng, country, reasoning, seconds):
        location = self.rounds[index]
        result = BenchmarkResult(
            location=location,
            guess=Guess(country=country.strip(), lat=lat, lng=lng),
        )
        result.calculate_metrics(self.scale)

        with self.lock:
            self.answers[location.id] = {
                "index": index,
                "lat": lat,
                "lng": lng,
                "country": country.strip(),
                "reasoning": reasoning.strip(),
                "seconds": round(seconds, 1),
                "km": result.distance_km,
                "score": result.score,
                "countryOk": bool(result.country_correct),
            }
            self.save()

        return self.reveal(index)

    def reveal(self, index):
        location = self.rounds[index]
        mine = self.answers[location.id]

        models = []
        for run in self.runs:
            guess = run["guesses"].get(location.id)
            if not guess:
                continue
            models.append({
                "name": run["name"],
                "provider": run["provider"],
                "text": reasoning_for(run, location.id),
                **guess,
            })
        models.sort(key=lambda m: (-(m["score"] or 0),
                                   m["km"] if m["km"] is not None else 1e9))

        beaten = sum(1 for m in models if (m["score"] or 0) < mine["score"])
        return {
            "index": index,
            "truth": {
                "id": location.id,
                "lat": location.lat,
                "lng": location.lng,
                "country": location.country,
            },
            "you": mine,
            "models": models,
            "beaten": beaten,
            "field": len(models),
        }

    # -- standings --

    def standings(self):
        ids = [loc.id for loc in self.rounds if loc.id in self.answers]
        if not ids:
            return {"rounds": [], "table": [], "played": 0, "ranked": 0,
                    "seed": self.seed, "runDir": str(self.run_dir)}

        rows = [self._row(self.name, "You", [self.answers[i] for i in ids], you=True)]
        for run in self.runs:
            # Only rank a run that played every location you did; a smoke-test
            # run over a single image is not a comparable score.
            if not all(i in run["guesses"] for i in ids):
                continue
            rows.append(self._row(run["name"], run["provider"],
                                  [run["guesses"][i] for i in ids]))
        rows.sort(key=lambda r: -r["points"])
        for rank, row in enumerate(rows, 1):
            row["rank"] = rank

        per_round = []
        for loc in self.rounds:
            if loc.id not in self.answers:
                continue
            mine = self.answers[loc.id]
            field = [(run["name"], run["guesses"][loc.id]) for run in self.runs
                     if loc.id in run["guesses"]]
            best = max(field, key=lambda f: f[1]["score"] or 0, default=None)
            per_round.append({
                "id": loc.id,
                "index": mine["index"],
                "country": loc.country,
                "km": mine["km"],
                "score": mine["score"],
                "countryOk": mine["countryOk"],
                "seconds": mine["seconds"],
                "bestName": best[0] if best else None,
                "bestScore": best[1]["score"] if best else None,
                "bestKm": best[1]["km"] if best else None,
            })

        return {
            "rounds": per_round,
            "table": rows,
            "played": len(ids),
            "ranked": len(rows) - 1,
            "seed": self.seed,
            "runDir": str(self.run_dir),
        }

    @staticmethod
    def _row(name, provider, guesses, you=False):
        scored = [g for g in guesses if not g.get("refused") and g.get("km") is not None]
        kms = sorted(g["km"] for g in scored)
        return {
            "name": name,
            "provider": provider,
            "you": you,
            "points": sum(g["score"] or 0 for g in guesses),
            "avgScore": round(sum(g["score"] or 0 for g in scored) / len(scored)) if scored else 0,
            "avgKm": round(sum(kms) / len(kms), 1) if kms else None,
            "medianKm": round(kms[len(kms) // 2], 1) if kms else None,
            "countries": sum(1 for g in guesses if g.get("countryOk")),
            "n": len(guesses),
        }

    # -- persistence: an ordinary run folder, rewritten after every round --

    def save(self):
        results = self.run_dir / "results"
        results.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "output").mkdir(parents=True, exist_ok=True)

        played = [loc for loc in self.rounds if loc.id in self.answers]
        records = []
        for loc in played:
            a = self.answers[loc.id]
            (self.run_dir / "output" / f"{loc.id}.txt").write_text(
                f"{a['reasoning']}\n\n"
                f"country: {a['country']}\nlat: {a['lat']}\nlng: {a['lng']}\n",
                encoding="utf-8",
            )
            records.append({
                "location_id": loc.id,
                "country_true": loc.country,
                "lat_true": loc.lat,
                "lng_true": loc.lng,
                "refused": False,
                "error_message": None,
                "country_guess": a["country"],
                "lat_guess": a["lat"],
                "lng_guess": a["lng"],
                "distance_km": a["km"],
                "score": a["score"],
                "country_correct": a["countryOk"],
            })
        pd.DataFrame(records).to_csv(results / "detailed.csv", index=False)

        kms = sorted(r["distance_km"] for r in records)
        scores = sorted(r["score"] for r in records)
        n = len(records)
        summary = {
            "model": self.name,
            "test": self.dataset,
            "n": n,
            "country_success_rate": sum(1 for r in records if r["country_correct"]) / n if n else 0,
            "refusal_rate": 0.0,
            "average_distance_km": sum(kms) / n if n else None,
            "average_score": sum(scores) / n if n else None,
            # Upper-middle median, the same one geobench.py writes.
            "median_distance_km": kms[n // 2] if n else None,
            "median_score": scores[n // 2] if n else None,
            "provider": "Human",
            "prompt_mode": "human",
        }
        (results / "summary.json").write_text(json.dumps(summary, indent=2),
                                              encoding="utf-8")

        (self.run_dir / "session.json").write_text(json.dumps({
            "dataset": self.dataset,
            "name": self.name,
            "seed": self.seed,
            "ids": [loc.id for loc in self.rounds],
            "answers": self.answers,
        }, indent=2), encoding="utf-8")


# --- serving ----------------------------------------------------------------

def page_payload(quiz):
    done = sorted(quiz.answers[loc.id]["index"] for loc in quiz.rounds
                  if loc.id in quiz.answers)
    return {
        "dataset": quiz.dataset,
        "name": quiz.name,
        "rounds": len(quiz.rounds),
        "pool": quiz.total_locations,
        "bounds": quiz.bounds,
        "scale": round(quiz.scale, 3),
        "models": len(quiz.runs),
        "countries": sorted(group[0] for group in COUNTRY_GROUPS),
        "runDir": str(quiz.run_dir),
        "done": done,
    }


def render(quiz):
    template = TEMPLATE.read_text(encoding="utf-8")
    if "/*__DATA__*/" not in template:
        raise SystemExit(f"{TEMPLATE} is missing the /*__DATA__*/ placeholder")
    blob = json.dumps(page_payload(quiz), ensure_ascii=False, separators=(",", ":"))
    return template.replace("/*__DATA__*/", blob.replace("</", "<\\/"))


class Handler(http.server.BaseHTTPRequestHandler):
    quiz = None
    verbose = False
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        if self.verbose:
            super().log_message(fmt, *args)

    def _send(self, body, ctype, status=200):
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, payload, status=200):
        self._send(json.dumps(payload).encode("utf-8"), "application/json", status)

    def do_GET(self):
        path = urlparse(self.path).path
        try:
            if path in ("/", "/index.html"):
                self._send(render(self.quiz).encode("utf-8"),
                           "text/html; charset=utf-8")
            elif path.startswith("/img/"):
                self._image(path[len("/img/"):])
            elif path == "/api/standings":
                self._json(self.quiz.standings())
            else:
                self._send(b"not found", "text/plain", 404)
        except Exception as exc:  # a broken round should not kill the server
            self._json({"error": str(exc)}, 500)

    def do_POST(self):
        path = urlparse(self.path).path
        try:
            length = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(length) or b"{}")
            if path == "/api/guess":
                self._json(self._guess(body))
            else:
                self._send(b"not found", "text/plain", 404)
        except Exception as exc:
            self._json({"error": str(exc)}, 500)

    def _image(self, raw):
        index = int(raw.split(".")[0])
        location = self.quiz.rounds[index]
        data = Path(location.image_path).read_bytes()
        ctype = mimetypes.guess_type(location.image_path)[0] or "image/jpeg"
        self._send(data, ctype)

    def _guess(self, body):
        index = int(body["round"])
        if not 0 <= index < len(self.quiz.rounds):
            raise ValueError(f"no such round: {index + 1}")
        lat, lng = float(body["lat"]), float(body["lng"])
        if not -90 <= lat <= 90 or not -180 <= lng <= 180:
            raise ValueError(f"coordinates out of range: {lat}, {lng}")
        country = str(body.get("country") or "").strip()
        reasoning = str(body.get("reasoning") or "").strip()
        if not country:
            raise ValueError("no country given")
        if not reasoning:
            raise ValueError("no reasoning given")

        reveal = self.quiz.submit(index, lat, lng, country, reasoning,
                                  float(body.get("seconds") or 0))
        loc = self.quiz.rounds[index]
        print(f"  {index + 1}/{len(self.quiz.rounds)}  #{loc.id}  "
              f"{reveal['you']['km']:,.0f} km  {reveal['you']['score']} pts  "
              f"beat {reveal['beaten']}/{reveal['field']} model runs", flush=True)
        return reveal


def serve(quiz, port, open_browser):
    server = http.server.ThreadingHTTPServer(("127.0.0.1", port), Handler)
    server.daemon_threads = True
    url = f"http://127.0.0.1:{port}/"
    print(f"  open {url}   (ctrl-c to stop; progress is saved every round)\n")
    if open_browser:
        threading.Thread(target=webbrowser.open, args=(url,), daemon=True).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print()
    finally:
        server.server_close()


# --- entry ------------------------------------------------------------------

def pick_rounds(locations, args):
    by_id = {loc.id: loc for loc in locations}
    if args.ids:
        missing = [i for i in args.ids if str(i) not in by_id]
        if missing:
            raise SystemExit("no such location(s) in the dataset: "
                             + ", ".join(map(str, missing)))
        return [by_id[str(i)] for i in args.ids], None
    if args.num > len(locations):
        raise SystemExit(f"asked for {args.num} locations, dataset has {len(locations)}")
    seed = args.seed if args.seed is not None else random.randrange(1_000_000)
    return random.Random(seed).sample(locations, args.num), seed


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-d", "--dataset", default="acw", help="dataset name (default: acw)")
    ap.add_argument("-n", "--num", type=int, default=10,
                    help="how many locations to play (default: 10)")
    ap.add_argument("-i", "--ids", nargs="*", default=None, metavar="ID",
                    help="play these location ids instead of a random sample")
    ap.add_argument("--seed", type=int, default=None,
                    help="sampling seed, printed at startup so a set can be replayed")
    ap.add_argument("--name", default=None,
                    help="what to call you in the results (default: $USER)")
    ap.add_argument("-m", "--models", nargs="*", default=None,
                    help="only compare against runs whose folder name contains these")
    ap.add_argument("--resume", type=Path, default=None, metavar="RUN",
                    help="continue a run folder written by an earlier session")
    ap.add_argument("--responses", type=Path, default=None,
                    help="responses directory (default: <repo>/responses)")
    ap.add_argument("--dataset-root", type=Path, default=None,
                    help="dataset directory root (default: <repo>/dataset)")
    ap.add_argument("--port", type=int, default=8030)
    ap.add_argument("--no-open", action="store_true", help="do not open a browser")
    ap.add_argument("-v", "--verbose", action="store_true", help="log every request")
    args = ap.parse_args()

    responses_dir = (args.responses or REPO_ROOT / "responses").resolve()
    dataset_root = (args.dataset_root or REPO_ROOT / "dataset").resolve()

    resumed = None
    if args.resume:
        run_dir = args.resume.resolve()
        resumed = read_json(run_dir / "session.json")
        if resumed is None:
            raise SystemExit(f"no session.json in {run_dir} - only quiz runs resume")
        args.dataset = resumed["dataset"]

    dataset_dir = dataset_root / args.dataset
    if not dataset_dir.is_dir():
        raise SystemExit(f"no dataset directory at {dataset_dir}")
    _, _, locations = load_dataset(dataset_dir)

    if resumed:
        by_id = {loc.id: loc for loc in locations}
        missing = [i for i in resumed["ids"] if i not in by_id]
        if missing:
            raise SystemExit("the resumed session names locations this dataset "
                             "no longer has: " + ", ".join(missing))
        rounds = [by_id[i] for i in resumed["ids"]]
        name, seed = resumed["name"], resumed.get("seed")
    else:
        rounds, seed = pick_rounds(locations, args)
        name = args.name or os.environ.get("USER") or "Human"
        stamp = datetime.datetime.now().strftime("%Y-%m-%dT%H_%M_%S")
        run_dir = responses_dir / f"{name}_{args.dataset}_{stamp}"

    quiz = Quiz(args.dataset, dataset_dir, responses_dir, rounds, name, run_dir,
                seed, args.models)
    if resumed:
        quiz.answers = resumed.get("answers", {})

    Handler.quiz = quiz
    Handler.verbose = args.verbose

    left = len(rounds) - len(quiz.answers)
    print(f"\n{name} vs {len(quiz.runs)} model runs on {args.dataset} "
          f"- {left} of {len(rounds)} rounds to play"
          + (f", seed {seed}" if seed is not None else ""))
    print(f"  locations: {', '.join(loc.id for loc in rounds)}")
    print(f"  writing:   {run_dir}")
    serve(quiz, args.port, not args.no_open)

    if quiz.answers:
        table = quiz.standings()["table"]
        you = next(r for r in table if r["you"])
        print(f"{you['points']:,} points over {you['n']} rounds "
              f"({you['avgKm']} km average, {you['countries']}/{you['n']} countries)"
              f" - rank {you['rank']} of {len(table)}")
        print(f"results in {run_dir}")


if __name__ == "__main__":
    main()
