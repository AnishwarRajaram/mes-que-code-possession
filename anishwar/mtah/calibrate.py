"""Calibrates the dead-ball rules on other matches (IDSSE), nested leave-one-match-out.

For each IDSSE match: cues from positions (deadball.build_cues), then the ball hidden the way our
tracking loses it (deadball.dropout, profile measured on our match's tracking, no labels), with
DROPOUT_SEEDS different random placements so no single placement decides anything.

Search, per fold, using only the six training matches:
  1. smart seeds: N_SEEDS spread-out points over RANGES (a Latin hypercube, fixed seed) plus the
     current PARAMS; the TOP_SEEDS best start a local search
  2. coordinate ascent: move one constant at a time by one or two steps while the training
     score improves
The fold's winner is then scored once on the held-out match: those held-out scores are the honest
estimate. The final constants are found the same way on all seven, and checked to sit on a
plateau: every one-step neighbour scores within PLATEAU of them.

Every score is cached per (constants, match), so the folds share work. Output is aggregate only.
"""

import os

import numpy as np

from . import config, deadball, external, load, shape

RANGES = {
    "SLOW": np.round(np.arange(0.50, 1.001, 0.05), 2),
    "RESUME": np.round(np.arange(0.80, 1.401, 0.05), 2),
    "MIN_STOP_S": np.arange(2.0, 12.01, 1.0),
    "START_BACK_S": np.arange(-2.0, 4.01, 0.5),
    "END_LEAD_S": np.arange(-1.0, 3.01, 0.5),
    "OUT_LINK_S": np.arange(4.0, 16.01, 2.0),
    "OUT_MIN_S": np.arange(2.0, 10.01, 1.0),
    "OUT_NEAR_M": np.array([0.0, 1.0, 2.0, 3.0, 5.0, 8.0]),
    "OUT_GAP_S": np.arange(0.5, 6.01, 0.5),
    "KO_SHARE": np.array([0.75, 0.8, 0.85, 0.9, 0.95, 1.01]),
    "KO_MIN_S": np.arange(0.5, 5.01, 0.5),
}
KEYS = list(RANGES)
DROPOUT_SEEDS = (0, 1, 2)
N_SEEDS = 40
TOP_SEEDS = 4
STEPS = (-2, -1, 1, 2)
PLATEAU = 0.01
CUES_VERSION = 5
SEED = 0


def our_cues():
    """Cues for the provided game's tactical tracking, and its graded mask."""
    atd = load.load_atd("tactical")
    team = atd["team"]
    on = (team == load.HOME) | (team == load.AWAY)
    ball = load.to_metres(atd["ball"][:, :2])
    players = load.to_metres(atd["players"][:, on, :2])
    n = len(ball)
    segment = (np.arange(n) >= config.half_blocks()[1][1]).astype(np.int8)
    t = (team[on] == load.AWAY).astype(np.int8)
    c = deadball.build_cues(ball, players, config.PITCH, config.FPS, segment, t)
    c["shape"] = shape.team_features(players, t, config.PITCH, config.FPS, segment)
    return c, load.graded_mask(n)


def idsse_cues(match_id):
    path = os.path.join(external.ARRAYS_DIR, match_id + "_cues%d.npz" % CUES_VERSION)
    if os.path.exists(path):
        with np.load(path) as z:
            c = {k: z[k] for k in z.files}
        c["fps"] = int(c["fps"])
        return c
    m = external.load(match_id)
    c = deadball.build_cues(m["ball_xy"], m["players_xy"], tuple(m["pitch"]), 25, m["period"],
                            m["team"])
    c["dead"] = ~m["alive"]
    c["owner"] = m["owner"]
    c["shape"] = shape.team_features(m["players_xy"], m["team"], tuple(m["pitch"]), 25,
                                     m["period"])
    np.savez_compressed(path, **c)
    return c


class Evaluator:
    """Per-match scores of a constants index vector, averaged over the dropout seeds, cached."""

    def __init__(self, match_ids, use_dropout):
        self.profile = deadball.ball_gap_profile(*our_cues()) if use_dropout else None
        self.variants = []                  # per match: list of cue dicts (one per seed)
        for i, mid in enumerate(match_ids):
            c = idsse_cues(mid)
            if use_dropout:
                self.variants.append([deadball.dropout(c, self.profile, seed=100 * i + s)
                                      for s in DROPOUT_SEEDS])
            else:
                self.variants.append([c])
        self.cache = {}

    def params(self, idx):
        return {k: float(RANGES[k][i]) for k, i in zip(KEYS, idx)}

    def valid(self, idx):
        p = self.params(idx)
        return p["SLOW"] < p["RESUME"]

    def scores(self, idx):
        """(n_matches, 4): f1_stop, f1_frame, dead share, stoppages per 90."""
        idx = tuple(int(i) for i in idx)
        if idx not in self.cache:
            p = self.params(idx)
            rows = []
            for vs in self.variants:
                s = [deadball.score(deadball.detect(c, p), c["dead"], c["segment"], c["fps"])
                     for c in vs]
                rows.append([np.mean([x[k] for x in s])
                             for k in ("f1_stop", "f1_frame", "dead_share", "stops_per90")])
            self.cache[idx] = np.array(rows)
        return self.cache[idx]

    def objective(self, idx, matches):
        return float(self.scores(idx)[matches, :2].mean())


def _seeds():
    rng = np.random.default_rng(SEED)
    sizes = [len(RANGES[k]) for k in KEYS]
    cols = [(rng.permutation(N_SEEDS) + rng.random(N_SEEDS)) / N_SEEDS for _ in KEYS]
    pts = [tuple(int(c[j] * n) for c, n in zip(cols, sizes)) for j in range(N_SEEDS)]
    current = tuple(int(np.abs(RANGES[k] - deadball.PARAMS[k]).argmin()) for k in KEYS)
    return [current] + pts


def _neighbours(idx, steps=STEPS):
    for j, k in enumerate(KEYS):
        for d in steps:
            v = idx[j] + d
            if 0 <= v < len(RANGES[k]):
                yield idx[:j] + (v,) + idx[j + 1:]


def search(ev, matches):
    """Best constants on these matches: top smart seeds, then coordinate ascent."""
    seeds = [s for s in _seeds() if ev.valid(s)]
    seeds.sort(key=lambda s: -ev.objective(s, matches))
    best, best_obj = None, -1.0
    for start in seeds[:TOP_SEEDS]:
        cur, cur_obj = start, ev.objective(start, matches)
        improved = True
        while improved:
            improved = False
            for nb in _neighbours(cur):
                if not ev.valid(nb):
                    continue
                o = ev.objective(nb, matches)
                if o > cur_obj + 1e-9:
                    cur, cur_obj, improved = nb, o, True
        if cur_obj > best_obj:
            best, best_obj = cur, cur_obj
    return best, best_obj


def run(match_ids=external.MATCHES, use_dropout=True):
    ev = Evaluator(match_ids, use_dropout)
    every = list(range(len(match_ids)))
    held_out = []
    for mi in every:
        train = [j for j in every if j != mi]
        best, _ = search(ev, train)
        held_out.append((match_ids[mi], ev.params(best), ev.scores(best)[mi]))
    best, obj = search(ev, every)
    edges = [k for j, k in enumerate(KEYS) if best[j] in (0, len(RANGES[k]) - 1)]
    nb = np.array([ev.objective(n, every) for n in _neighbours(best, (-1, 1)) if ev.valid(n)])
    truth = []
    for vs in ev.variants:
        c = vs[0]
        a, b = deadball.runs(c["dead"])
        truth.append((c["dead"].mean(), ((b - a) >= c["fps"]).sum() * 90 / (len(c["dead"]) / 1500)))
    return {"profile": ev.profile, "held_out": held_out, "best": ev.params(best),
            "best_objective": obj, "neighbours": nb, "edges": edges,
            "evaluated": len(ev.cache), "truth": np.array(truth).mean(axis=0)}


def _fmt(p):
    return " ".join("{}={:g}".format(k, v) for k, v in p.items())


def report(use_dropout=True):
    r = run(use_dropout=use_dropout)
    lines = ["IDSSE, ball {}{}; {} constant sets evaluated".format(
        "hidden like ours" if use_dropout else "fully tracked",
        " ({} placements averaged)".format(len(DROPOUT_SEEDS)) if use_dropout else "",
        r["evaluated"]),
        "", "nested leave-one-match-out (search on the other six, scored once here):",
        "  match    stopF1  frameF1  dead%  stops/90   chosen"]
    for mid, p, s in r["held_out"]:
        lines.append("  {}   {:.3f}   {:.3f}   {:4.1f}  {:5.0f}     {}".format(
            mid, s[0], s[1], 100 * s[2], s[3], _fmt(p)))
    h = np.array([s for _, _, s in r["held_out"]])
    lines += ["  mean     {:.3f}   {:.3f}   {:4.1f}  {:5.0f}".format(
        h[:, 0].mean(), h[:, 1].mean(), 100 * h[:, 2].mean(), h[:, 3].mean()),
        "  sd       {:.3f}   {:.3f}".format(h[:, 0].std(), h[:, 1].std()),
        "  truth                      {:4.1f}  {:5.0f}".format(100 * r["truth"][0],
                                                             r["truth"][1]),
        "", "best on all seven: " + _fmt(r["best"]),
        "  objective {:.3f}; {} one-step neighbours: min {:.3f}, median {:.3f}; "
        "within {}: {}/{}".format(r["best_objective"], len(r["neighbours"]),
                                   r["neighbours"].min(), np.median(r["neighbours"]), PLATEAU,
                                   int((r["neighbours"] >= r["best_objective"] - PLATEAU).sum()),
                                   len(r["neighbours"])),
        "  at the edge of its range: " + (", ".join(r["edges"]) or "none")]
    return "\n".join(lines)
