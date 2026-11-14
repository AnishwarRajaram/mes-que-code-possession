"""Calibrates the possession rules end to end on other matches (IDSSE), nested leave-one-match-out.

The whole pipeline runs as it will on our match: the ball hidden the way our tracking loses it,
dead ball from our own dead-ball rules (deadball.PARAMS, already calibrated), possession from
possession.label. The target is the DFL's dead ball and owner through the same control filter
(possession.target). Scored with the organisers' own metric code (evaluation/possession_metrics):
spellF1, which ranks, is the objective; macroF1 is reported.

Search as in calibrate.py: smart seeds (Latin hypercube + current PARAMS), the best few refined by
coordinate ascent on the six training matches, scored once on the held-out one. Output is
aggregate only. Also reported: the same pipeline with the true dead ball and with the ball fully
tracked, which splits the remaining error between dead ball, ball loss and possession rules.
"""

import numpy as np

from . import calibrate, deadball, external, harness, possession

pm = harness.pm
RANGES = {
    "TOUCH_R": np.arange(0.5, 3.01, 0.25),
    "MARGIN_M": np.arange(0.0, 1.51, 0.25),
    "TURN_DEG": np.array([0.0, 5.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0]),
    "JUMP_MS": np.arange(1.0, 14.01, 1.0),
    "CARRY_MS": np.arange(0.0, 6.01, 1.0),
    "GAP_FRAC": np.arange(0.0, 1.001, 0.25),
}
KEYS = list(RANGES)
N_SEEDS = 30
TOP_SEEDS = 3
STEPS = (-2, -1, 1, 2)
PLATEAU = 0.01
SEED = 0


class Evaluator:
    def __init__(self, match_ids, use_dropout=True, true_dead=False):
        profile = deadball.ball_gap_profile(*calibrate.our_cues()) if use_dropout else None
        self.items = []                          # per match: list of (cues, dead, motion, gt, runs)
        for i, mid in enumerate(match_ids):
            c = calibrate.idsse_cues(mid)
            gt = possession.target(c).copy()
            gt[np.flatnonzero(np.diff(c["segment"]) != 0) + 1] = harness.X   # split the halves
            runs = pm.valid_runs(gt)
            variants = [deadball.dropout(c, profile, seed=100 * i + s)
                        for s in calibrate.DROPOUT_SEEDS] if use_dropout else [c]
            self.items.append([(v, c["dead"] if true_dead else deadball.detect(v),
                                possession.ball_motion(v), gt, runs) for v in variants])
        self.cache = {}

    def params(self, idx):
        return {k: float(RANGES[k][i]) for k, i in zip(KEYS, idx)}

    def scores(self, idx):
        """(n_matches, 4): spellF1, macroF1, possessions per 90 predicted, and in the target."""
        idx = tuple(int(i) for i in idx)
        if idx not in self.cache:
            p = self.params(idx)
            rows = []
            for vs in self.items:
                s = []
                for v, dead, motion, gt, runs in vs:
                    pred = possession.label(v, dead, p, motion)
                    pred[gt == harness.X] = harness.X
                    minutes = len(gt) / v["fps"] / 60
                    s.append((pm.spell_scores(gt, pred, runs)["mean_f1"],
                              pm.macro_f1(pm.scored_confusion(pm.confusion(gt, pred))),
                              len(pm.possession_spells(pred, runs)) * 90 / minutes,
                              len(pm.possession_spells(gt, runs)) * 90 / minutes))
                rows.append(np.mean(s, axis=0))
            self.cache[idx] = np.array(rows)
        return self.cache[idx]

    def objective(self, idx, matches):
        return float(self.scores(idx)[matches, 0].mean())


def _seeds():
    rng = np.random.default_rng(SEED)
    sizes = [len(RANGES[k]) for k in KEYS]
    cols = [(rng.permutation(N_SEEDS) + rng.random(N_SEEDS)) / N_SEEDS for _ in KEYS]
    pts = [tuple(int(c[j] * n) for c, n in zip(cols, sizes)) for j in range(N_SEEDS)]
    # Only the fixed-seed points: the stored constants are never a starting point, so the search
    # depends on the code and the IDSSE data alone and re-running it gives back what is stored.
    return pts


def _neighbours(idx, steps=STEPS):
    for j, k in enumerate(KEYS):
        for d in steps:
            v = idx[j] + d
            if 0 <= v < len(RANGES[k]):
                yield idx[:j] + (v,) + idx[j + 1:]


def search(ev, matches):
    seeds = sorted(_seeds(), key=lambda s: -ev.objective(s, matches))
    best, best_obj = None, -1.0
    for start in seeds[:TOP_SEEDS]:
        cur, cur_obj = start, ev.objective(start, matches)
        improved = True
        while improved:
            improved = False
            for nb in _neighbours(cur):
                o = ev.objective(nb, matches)
                if o > cur_obj + 1e-9:
                    cur, cur_obj, improved = nb, o, True
        if cur_obj > best_obj:
            best, best_obj = cur, cur_obj
    return best, best_obj


def run(match_ids=external.MATCHES):
    ev = Evaluator(match_ids)
    every = list(range(len(match_ids)))
    sets = [[j for j in every if j != mi] for mi in every] + [every]
    found = calibrate.fold_searches(search, ev, sets)
    held = [(match_ids[mi], ev.params(found[mi][0]), ev.scores(found[mi][0])[mi]) for mi in every]
    best, obj = found[-1]
    nb = np.array([ev.objective(n, every) for n in _neighbours(best, (-1, 1))])
    edges = [k for j, k in enumerate(KEYS) if best[j] in (0, len(RANGES[k]) - 1)]
    split = {}
    for name, kw in (("true dead ball, ball hidden", dict(true_dead=True)),
                     ("our dead ball, ball fully tracked", dict(use_dropout=False)),
                     ("true dead ball, ball fully tracked", dict(use_dropout=False,
                                                                 true_dead=True))):
        split[name] = Evaluator(match_ids, **kw).scores(best).mean(axis=0)
    return {"held": held, "best": ev.params(best), "obj": obj, "neighbours": nb, "edges": edges,
            "split": split, "evaluated": len(ev.cache), "all": ev.scores(best).mean(axis=0)}


def report(r=None):
    r = run() if r is None else r
    fmt = lambda p: " ".join("{}={:g}".format(k, v) for k, v in p.items())  # noqa: E731
    lines = ["possession, IDSSE end to end (our dead ball, ball hidden like ours); "
             "{} constant sets evaluated".format(r["evaluated"]), "",
             "nested leave-one-match-out:", "  match    spellF1  macroF1  poss/90 (target)  chosen"]
    for mid, p, s in r["held"]:
        lines.append("  {}   {:.3f}    {:.3f}    {:4.0f} ({:3.0f})     {}".format(
            mid, s[0], s[1], s[2], s[3], fmt(p)))
    h = np.array([s for _, _, s in r["held"]])
    lines += ["  mean     {:.3f}    {:.3f}    {:4.0f} ({:3.0f})".format(*h.mean(axis=0)),
              "  sd       {:.3f}    {:.3f}".format(h[:, 0].std(), h[:, 1].std()), "",
              "best on all seven: " + fmt(r["best"]),
              "  spellF1 {:.3f}; {} one-step neighbours: min {:.3f}, median {:.3f}; within {}: "
              "{}/{}".format(r["obj"], len(r["neighbours"]), r["neighbours"].min(),
                             np.median(r["neighbours"]), PLATEAU,
                             int((r["neighbours"] >= r["obj"] - PLATEAU).sum()),
                             len(r["neighbours"])),
              "  at the edge of its range: " + (", ".join(r["edges"]) or "none"), "",
              "where the error comes from (same constants, all seven):",
              "  {:<38} spellF1 {:.3f}  macroF1 {:.3f}".format("as submitted (our dead, hidden)",
                                                               *r["all"][:2])]
    for name, s in r["split"].items():
        lines.append("  {:<38} spellF1 {:.3f}  macroF1 {:.3f}".format(name, s[0], s[1]))
    return "\n".join(lines)


# ---- Rule 6: the players' shape inside ball gaps ---------------------------------------------

SHAPE_GRID = {"SWITCH_P": (0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.8),
              "SWITCH_S": (0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0),
              "SMOOTH_S": (0.0,)}
SHAPE_HZ = 5


def shape_inputs(v, codes0):
    from . import shape
    return shape.inputs(v, codes0)


def shape_fit(match_ids=external.MATCHES):
    """The shape model on all seven matches, as stored in shape.MODEL."""
    from . import shape
    ev = Evaluator(match_ids)
    Xs, ys = [], []
    for vs in ev.items:
        for v, dead, motion, gt, runs in vs:
            X = shape.inputs(v, possession.label(v, dead, possession.PARAMS, motion))
            mask = ~v["ball_seen"] & ~dead & np.isin(gt, (0, 1))
            mask[np.arange(len(mask)) % (v["fps"] // SHAPE_HZ) != 0] = False
            Xs.append(X[mask])
            ys.append((gt[mask] == possession.HOME).astype(float))
    return shape.fit(np.concatenate(Xs), np.concatenate(ys))


def shape_run(match_ids=external.MATCHES):
    import itertools
    from . import shape
    ev = Evaluator(match_ids)
    p = possession.PARAMS
    prepared = []                               # per match: list of (v, dead, motion, gt, runs, X, codes0)
    for vs in ev.items:
        rows = []
        for v, dead, motion, gt, runs in vs:
            codes0 = possession.label(v, dead, p, motion)
            rows.append((v, dead, motion, gt, runs, shape_inputs(v, codes0), codes0))
        prepared.append(rows)
    combos = [dict(zip(SHAPE_GRID, c)) for c in itertools.product(*SHAPE_GRID.values())]

    def spell(row, p_home, q):
        v, dead, motion, gt, runs, X, _ = row
        pred = possession.label(v, dead, p, motion, p_home, q)
        pred[gt == harness.X] = harness.X
        return (pm.spell_scores(gt, pred, runs)["mean_f1"],
                pm.macro_f1(pm.scored_confusion(pm.confusion(gt, pred))))

    def train(ms):
        Xs, ys = [], []
        for m in ms:
            for v, dead, motion, gt, runs, X, _ in prepared[m]:
                mask = ~v["ball_seen"] & ~dead & np.isin(gt, (0, 1))
                mask[np.arange(len(mask)) % (v["fps"] // SHAPE_HZ) != 0] = False
                Xs.append(X[mask])
                ys.append((gt[mask] == possession.HOME).astype(float))
        model = shape.fit(np.concatenate(Xs), np.concatenate(ys))
        probs = {m: [shape.prob(model, row[5]) for row in prepared[m]] for m in ms}
        best = max(combos, key=lambda q: np.mean([spell(row, ph, q)[0] for m in ms
                                                  for row, ph in zip(prepared[m], probs[m])]))
        return model, best

    every = list(range(len(match_ids)))
    held, base, acc, hold_acc = [], [], [], []
    for mi in every:
        model, q = train([j for j in every if j != mi])
        s = [spell(row, shape.prob(model, row[5]), q) for row in prepared[mi]]
        b = [spell(row, None, q) for row in prepared[mi]]
        held.append((match_ids[mi], q, np.mean(s, axis=0)))
        base.append(np.mean(b, axis=0))
        for v, dead, motion, gt, runs, X, codes0 in prepared[mi]:
            mask = ~v["ball_seen"] & ~dead & np.isin(gt, (0, 1))
            ph = shape.prob(model, X[mask])
            acc.append(((ph >= 0.5) == (gt[mask] == possession.HOME)).mean())
            hold_acc.append((codes0[mask] == gt[mask]).mean())
    model, q = train(every)
    return {"held": held, "base": np.array(base), "q": q, "model": model,
            "acc": float(np.mean(acc)), "hold_acc": float(np.mean(hold_acc))}


def shape_report(r=None):
    from . import shape
    r = shape_run() if r is None else r
    lines = ["rule 6 (players' shape inside ball gaps), IDSSE end to end, nested leave-one-match-out",
             "  match    spellF1 (without)  macroF1 (without)   chosen"]
    for (mid, q, s), b in zip(r["held"], r["base"]):
        lines.append("  {}   {:.3f} ({:.3f})      {:.3f} ({:.3f})       {}".format(
            mid, s[0], b[0], s[1], b[1], " ".join("%s=%g" % kv for kv in q.items())))
    h = np.array([s for _, _, s in r["held"]])
    lines += ["  mean     {:.3f} ({:.3f})      {:.3f} ({:.3f})".format(
        h[:, 0].mean(), r["base"][:, 0].mean(), h[:, 1].mean(), r["base"][:, 1].mean()),
        "  owner accuracy on hidden-ball live frames, held out: shape model {:.1%}, holding the "
        "last owner {:.1%}".format(r["acc"], r["hold_acc"]),
        "", "fitted on all seven: " + " ".join("%s=%g" % kv for kv in r["q"].items()),
        "  standardised weights (towards home having the ball):"]
    mu, sd, w = r["model"]
    for name, x in sorted(zip(("bias",) + shape.NAMES, w), key=lambda t: -abs(t[1])):
        lines.append("    {:<32} {:+.2f}".format(name, x))
    return "\n".join(lines)
