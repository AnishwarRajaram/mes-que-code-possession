"""Builds the submission file for the provided game from the tracking alone.

    cues (deadball.build_cues) -> dead ball (deadball.detect) -> possession (possession.label)
    -> players' shape inside ball gaps (shape.MODEL) -> possession again with rule 6

Every constant comes from deadball.PARAMS and possession.PARAMS, calibrated on other matches.
Graded frames get H, A or D (never X: on spellF1 an X costs what repeating the last label costs,
and on macroF1 it is a miss); ungraded frames get X. Deterministic: the same data gives the same
file, byte for byte.
"""

import hashlib
import os

import numpy as np

from . import calibrate, config, deadball, harness, possession, shape


def codes():
    cues, graded = calibrate.our_cues()
    dead = deadball.detect(cues)
    first = possession.label(cues, dead)
    p_home = shape.prob(shape.MODEL, shape.inputs(cues, first))
    out = possession.label(cues, dead, p_home=p_home).astype(np.int8)  # 0 H, 1 A, 2 D
    out[~graded] = harness.X
    return out


def write(name):
    c = codes()
    labels = np.array(harness.pm.LABELS)[c]
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    path = os.path.join(config.RESULTS_DIR, name + ".csv")
    with open(path, "w", newline="\n") as f:
        f.write("frame,label\n")
        f.write("".join("{},{}\n".format(i, l) for i, l in enumerate(labels)))
    with open(path, "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()
    return path, digest, c



