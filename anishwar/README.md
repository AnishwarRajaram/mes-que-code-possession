# anishwar/ — Mes Que Code possession pipeline

Working folder for team Mes Que Code. Infrastructure only so far: loading, features, scoring and
label-free checks. The possession rules are being rebuilt.

## Rules we work under

From the organisers' [RULES.md](../RULES.md), [FAQ.md](../FAQ.md) and the terms of use:

- Every label comes from an algorithm. No hand labels or hand corrections on the provided game,
  for any purpose, including training or validation.
- The public 5 minutes are a correctness check only. They are not training data and not a tuning
  set, and they never choose between versions.
- No human-produced data about the provided game from any source. Data about other matches,
  including hand labels, is fine and is documented.
- No rules or constants tied to specific frames of the provided game.
- The match data never goes to a hosted AI, and nothing derived from it is committed.

## Setup

Tested with Python 3.12.8, numpy 1.26.3 and pandas 2.1.4.

```bash
pip install -r anishwar/requirements.txt
```

Data goes in the repository's `data/` folder (or `MTAH_DATA_DIR`), as in the root `DATA.md`.

## Commands

```bash
python anishwar/pipeline.py status          # environment, paths, which assets are present
python anishwar/pipeline.py cache           # parse the tactical tracking and tagging once
python anishwar/pipeline.py features        # per-frame features, full-match summary
python anishwar/pipeline.py census CSV...   # label-free plausibility checks
python anishwar/pipeline.py eval CSV...     # correctness check on the public window
```

All output is aggregate-only. Everything the pipeline writes (`cache/`, `results/`, `eval_out/`)
is gitignored, because it is derived from the match data.
