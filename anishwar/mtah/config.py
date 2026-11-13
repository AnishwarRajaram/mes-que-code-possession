"""Paths and constants for the pipeline.

Match facts (frame counts, halves, pitch, asset paths) come from the repository's games.py, so
there is one source of truth for them. This module only adds where *our* outputs go, all of
them inside anishwar/ and all of them gitignored.
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
WORKSPACE = os.path.dirname(HERE)
REPO_ROOT = os.path.dirname(WORKSPACE)

if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import games  # noqa: E402  (repo root, standard library only)

GAME = games.GAME
N_FRAMES = games.GAMES[GAME]["n_video_frames"]
FPS = games.GAMES[GAME]["fps"]
PITCH = games.GAMES[GAME]["pitch"]  # metres, (length, width)

# Our outputs. Everything here is derived from the match data and is gitignored.
CACHE_DIR = os.path.join(WORKSPACE, "cache")
RESULTS_DIR = os.path.join(WORKSPACE, "results")
EVAL_OUT_DIR = os.path.join(WORKSPACE, "eval_out")

# The repository's scorer and public ground truth, used as they are.
EVALUATION_DIR = os.path.join(REPO_ROOT, "evaluation")
PUBLIC_GT = os.path.join(EVALUATION_DIR, "gt", "{}_possession_gt_public.csv".format(GAME))


def asset_paths(source="tactical"):
    """games.asset_paths, re-exported so callers import one module."""
    return games.asset_paths(source=source)


def half_blocks():
    """[(half, first_video_frame, last_video_frame), ...] for the graded halves."""
    return games.half_video_blocks(GAME)
