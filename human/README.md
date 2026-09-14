# Play it yourself

The benchmark, but you are the model: a handful of the dataset's locations, your
own pin, your own written reasoning, scored with the same GeoGuessr formula and
saved in the same run format the models produce.

```bash
python human/quiz.py                        # 10 random acw locations
python human/quiz.py -d poland -n 5         # five from another dataset
python human/quiz.py --ids 3 7 12           # play specific locations
python human/quiz.py --seed 1234            # replay someone else's ten
python human/quiz.py --resume responses/julian_acw_2026-09-14T10_00_00
```

It serves a page on `127.0.0.1:8030` and opens a browser.

## A round

The image on the left, the map on the right. Drop a pin (or type coordinates —
the fields and the pin follow each other), name the country, and write why. All
three are required: no reasoning, no submit.

Submitting reveals the round — the true point joined to your pin, your error and
score, and then **every model run in `responses/` that saw the same image**,
ranked, with you slotted into the table where your score puts you. Click any row
to read that model's own reasoning for its guess.

After the last round you get your totals and the standings over exactly the
locations you played. Only runs that answered all of them are ranked; a
one-image smoke-test run is not a comparable score.

Keys: `ctrl`/`⌘ + enter` submit · `enter` next round · `f` zoom the image ·
`esc` close it.

**Nothing is leaked.** The page is handed the dataset name, the bounds and a
list of country names — never the answers. The truth, the scoring and the other
models' guesses live on the server and are sent only once a round has been
submitted.

## What it writes

An ordinary run folder, rewritten after every round:

```
responses/<name>_<dataset>_<stamp>/
  output/<id>.txt          your reasoning + the country/lat/lng block
  results/detailed.csv     the same columns geobench.py writes
  results/summary.json     model: <name>, provider: Human, prompt_mode: human
  session.json             which locations, and everything you typed
```

So the rest of the repo already understands it:

```bash
python visualizations/viewer/build.py -d acw      # you, beside the models
python -m scripts.parser responses/<run>          # re-parse your answers
python -m scripts.fix responses/<run>/results/detailed.csv -m dataset/acw/metadata.json
```

Scores come from `geobench.calculate_score` through `BenchmarkResult`, the same
call the benchmark makes, so `scripts/fix.py` recomputes your CSV to the byte.

Quit any time with ctrl-c — every round is already on disk, and `--resume <run
folder>` picks up at the first unplayed location.

## Options

| | |
|---|---|
| `-d`, `--dataset` | dataset name (default `acw`) |
| `-n`, `--num` | how many locations (default 10) |
| `-i`, `--ids` | play these ids instead of a random sample |
| `--seed` | sampling seed; printed at startup so a set can be replayed |
| `--name` | what to call you in the results (default `$USER`) |
| `-m`, `--models` | only compare against runs whose folder name contains these |
| `--resume` | continue a run folder from an earlier session |
| `--responses`, `--dataset-root` | point at other trees |
| `--port`, `--no-open`, `-v` | serving |

## Notes

- **Over SSH** — the server binds to loopback only, so forward the port:
  `ssh -L 8030:127.0.0.1:8030 <host>`, then open the URL on your own machine.
  Add `--no-open` to stop it reaching for a browser on the remote box.
- Scores are only comparable within one dataset — the GeoGuessr scale comes from
  that map's bounds. The scale is printed in the payload and the standings table
  is always over one dataset.
- The page re-renders from `template.html` on every request, so editing the
  template and refreshing is enough; no build step.
- Tiles and web fonts want a network connection. Without one the page still
  plays: if Leaflet fails to load the map is replaced by a note and the latitude
  and longitude fields take the guess.
