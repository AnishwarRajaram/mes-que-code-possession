"""SkillCorner open data (A-League 2024/25, MIT licence, credit SkillCorner) as a check of the
camera-off-the-pitch rule of broadcast.py. Aggregates only.

The full pipeline does not transfer to this source: 10 fps, about a third of the players tracked
from broadcast video, and dead ball known only as the gaps between possession phases. The check is
therefore at the level of the rule alone, fixed in advance:

- in play: frames inside a match period; dead: in play and outside every phase (survey choice S2);
- off-pitch: at most 2 players detected, as on the provided broadcast when its homography fails;
- the rule: off-pitch runs of at least broadcast.CAMERA_MIN_S seconds are dead.

Reported per match: off-pitch share of dead and of live time, the share of rule frames that are
labelled dead (precision), and the share of labelled dead the rule covers (recall). The data is
read from the folder named by the environment variable SC_DIR; nothing derived is committed.
"""

import json
import os

import numpy as np

from . import broadcast, deadball

SC_DIR = os.environ.get("SC_DIR", "")
FPS = 10
OFF_PLAYERS = 2


def match_ids():
    return sorted(f.split("_")[0] for f in os.listdir(os.path.join(SC_DIR, "tracking")) if f.endswith(".jsonl"))


def frames(mid):
    """Per frame index: in a period, inside a phase, detected player count."""
    meta = json.load(open(os.path.join(SC_DIR, "meta", "%s_match.json" % mid)))
    n = max(p["end_frame"] for p in meta["match_periods"]) + 1
    play = np.zeros(n, bool)
    for p in meta["match_periods"]:
        play[p["start_frame"]:p["end_frame"] + 1] = True
    phase = np.zeros(n, bool)
    import csv
    with open(os.path.join(SC_DIR, "phases", "%s_phases_of_play.csv" % mid)) as f:
        for r in csv.DictReader(f):
            phase[int(float(r["frame_start"])):int(float(r["frame_end"])) + 1] = True
    det = np.full(n, -1, np.int16)
    with open(os.path.join(SC_DIR, "tracking", "%s_tracking_extrapolated.jsonl" % mid)) as f:
        for line in f:
            d = json.loads(line)
            fr = d["frame"]
            if 0 <= fr < n:
                det[fr] = sum(1 for p in d["player_data"] or [] if p.get("is_detected") and p.get("x") is not None)
    return play, phase, det


def one(mid):
    play, phase, det = frames(mid)
    seen = play & (det >= 0)
    dead = seen & ~phase
    live = seen & phase
    off = seen & (det <= OFF_PLAYERS)
    rule = broadcast.add_offpitch(np.zeros_like(off), off, np.zeros(len(off)), broadcast.CAMERA_MIN_S,
                                  broadcast.CAMERA_MAX_RATIO, fps=FPS) & seen
    return (mid, float(off[dead].mean()), float(off[live].mean()), float(dead[rule].mean()),
            float(rule[dead].mean()), float(dead.sum() / seen.sum()))


if __name__ == "__main__":
    from multiprocessing import Pool
    with Pool(4) as pool:
        rows = pool.map(one, match_ids())
    print("match     off-pitch in dead / live   rule precision   rule recall   labelled dead share")
    for r in rows:
        print("  %-8s %.2f / %.2f                %.2f             %.2f          %.2f" % r)
    a = np.array([r[1:] for r in rows])
    print("  mean     %.2f / %.2f                %.2f             %.2f          %.2f" % tuple(a.mean(axis=0)))
