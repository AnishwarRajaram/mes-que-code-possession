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

## Data from other matches

Calibration uses the IDSSE dataset: seven Bundesliga 2022/23 matches with official DFL position
data and per-frame ball status and possession. Bassek, M., Rein, R., Weber, H. & Memmert, D.
(2025). An integrated dataset of spatiotemporal and event data in elite soccer. *Scientific Data*
12, 195. https://doi.org/10.1038/s41597-025-04505-y. Data © DFL, released under CC-BY 4.0. It is
downloaded from the dataset's public mirror (huggingface.co/datasets/pysport/idsse-data) into
`anishwar/external/idsse/` and read with kloppy; it is never committed.

```bash
python anishwar/pipeline.py cues            # dead-ball cues measured on the IDSSE matches
python anishwar/pipeline.py calibrate       # fit the dead-ball rules, nested leave-one-match-out
python anishwar/pipeline.py challenger      # logistic regression challenger under the same test
```

Before scoring, the IDSSE ball is hidden the way our tactical tracking loses it: in long gaps,
more often while players are slow. That profile is measured on our tracking alone, with no
labels. Both commands run single-threaded in a few minutes and under 1 GB of memory.

All output is aggregate-only. Everything the pipeline writes (`cache/`, `results/`, `eval_out/`)
is gitignored, because it is derived from the match data.
