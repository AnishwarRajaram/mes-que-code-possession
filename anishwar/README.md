# Més Que Code: Possession Pipeline

This folder holds the pipeline of team Més Que Code for More than a Hack 2027, Phase 1. The pipeline builds the submission file from the automatic tactical tracking. The calibration on other matches sets its constants. The technical document [Més Que Code.md](M%C3%A9s%20Que%20Code.md) describes the method. The guide [AGENTS.md](AGENTS.md) describes how the work is conducted and reproduced.

## 1. Rules Observed

The pipeline follows the organisers' [RULES.md](../RULES.md), [FAQ.md](../FAQ.md) and terms of use:

- Every label comes from an algorithm. No hand label or hand correction on the provided match is used for any purpose, including training and validation.
- The public five-minute window serves as a correctness check only. It never trains, tunes or selects between versions.
- No human-produced data about the provided match is used. Data about other matches is documented below.
- No rule or constant refers to a specific frame of the provided match.
- No match data goes to a hosted language model. Nothing derived from the match data is committed.

## 2. Setup

The pipeline is tested with Python 3.12.8, numpy 1.26.3, pandas 2.1.4 and kloppy 3.19.1.

```bash
pip install -r anishwar/requirements.txt
```

The provided data goes in the repository's `data/` folder, or in the folder named by `MTAH_DATA_DIR`, as described in the root `DATA.md`.

## 3. Commands

```bash
python anishwar/pipeline.py status          # environment, paths and assets present
python anishwar/pipeline.py cache           # parse the tactical tracking and tagging once
python anishwar/pipeline.py features        # per-frame features and a full-match summary
python anishwar/pipeline.py census CSV...   # label-free plausibility checks
python anishwar/pipeline.py deadcheck       # dead-ball rules on the provided match, label-free
python anishwar/pipeline.py run             # build the submission, results/Més Que Code.csv
python anishwar/pipeline.py eval CSV...     # correctness check on the public window
```

The build is deterministic. It requires only the provided data and writes `anishwar/results/Més Que Code.csv`.

## 4. Data from Other Matches

Calibration uses the IDSSE dataset: seven Bundesliga 2022/23 matches with official DFL position data and per-frame ball status and possession.

Bassek, M., Rein, R., Weber, H. and Memmert, D. (2025). An integrated dataset of spatiotemporal and event data in elite soccer. *Scientific Data* 12, 195. https://doi.org/10.1038/s41597-025-04505-y. Data © DFL, released under CC-BY 4.0.

The data is downloaded from the public mirror huggingface.co/datasets/pysport/idsse-data into `anishwar/external/idsse/` and read with kloppy. It is never committed. The following commands re-derive the constants in order:

```bash
python anishwar/pipeline.py cues            # dead-ball cues measured on the IDSSE matches
python anishwar/pipeline.py smoothing       # smoothing that matches the provided tracking's wobble
python anishwar/pipeline.py calibrate       # dead-ball rules, nested leave-one-match-out
python anishwar/pipeline.py possession      # possession rules on the stored dead-ball constants
python anishwar/pipeline.py changes         # change detector and team-shape model
python anishwar/pipeline.py challenger      # logistic challenger under the same test
```

Before scoring, the IDSSE tracking is adjusted to resemble the provided tracking. Its tracks are smoothed until their frame-to-frame wobble matches. Its ball is hidden in the pattern measured on the provided tracking: long gaps, more often while players are slow. Both adjustments are measured on the provided tracking alone, without labels. Each command runs single-threaded by default. The option `--workers N` searches N folds side by side with an identical result.

The broadcast camera rule is chosen on PFF FC World Cup 2022 (request-gated, no redistribution, processed locally only) and checked on SkillCorner open data (MIT licence, credit SkillCorner). Their folders are named by the environment variables `PFF_DIR` and `SC_DIR`. Run from the `anishwar/` folder:

```bash
python -m mtah.broadcast align                  # broadcast synchronisation, label-free checks
python -m mtah.pff read                         # then smoothing, sanity, camera, breakdown
python -m mtah.skillcorner                      # precision of the camera rule
```

The nested searches ran once on a private cloud CPU notebook. The folder `offload/` holds the scripts and the notebook. Section 8 of the technical document describes how the results were verified locally.

## 5. Outputs

Every command prints aggregates only. Everything the pipeline writes (`cache/`, `results/`, `eval_out/`) is excluded from version control because it derives from the match data.
