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
        profile = calibrate.hiding_profile() if use_dropout else None
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
# Tested and rejected (held-out spellF1 -0.001, better on 2 of 7): kept off, reproducible.
SHAPE_PRESS = False      # include the pressure features (shape.pressure_features) in the model
# Rule 6 as a hidden Markov model (possession.viterbi_override): the cost of a change of hands,
# in log-odds seconds of shape evidence. Used instead of SHAPE_GRID when SHAPE_DECODER is "viterbi".
# Tested: held-out spellF1 0.532 vs 0.531 for the threshold rule, better on 3 of 7 -> not adopted.
HMM_GRID = {"SWITCH_COST": (0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0)}
SHAPE_DECODER = "threshold"


def shape_inputs(v, codes0):
    from . import shape
    return shape.inputs(v, codes0, press=SHAPE_PRESS)


def shape_fit(match_ids=external.MATCHES):
    """The shape model on all seven matches, as stored in shape.MODEL."""
    from . import shape
    ev = Evaluator(match_ids)
    Xs, ys = [], []
    for vs in ev.items:
        for v, dead, motion, gt, runs in vs:
            X = shape_inputs(v, possession.label(v, dead, possession.PARAMS, motion))
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
    grid = HMM_GRID if SHAPE_DECODER == "viterbi" else SHAPE_GRID
    combos = [dict(zip(grid, c)) for c in itertools.product(*grid.values())]

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
    for name, x in sorted(zip(("bias",) + shape.names(len(w) - 1), w), key=lambda t: -abs(t[1])):
        lines.append("    {:<32} {:+.2f}".format(name, x))
    return "\n".join(lines)


# ---- owner model over all live play (owner.py) -------------------------------------------------
OWNER_GRID = (0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0)   # SWITCH_COST, log-odds seconds


def owner_run(match_ids=external.MATCHES):
    """Nested leave-one-match-out: the owner model is fitted, and its switch cost chosen, on six
    matches and scored once on the seventh, end to end with our dead ball and the ball hidden."""
    from . import owner
    ev = Evaluator(match_ids)
    prepared = []                                # per match: (v, dead, gt, runs, X, codes0)
    for vs in ev.items:
        rows = []
        for v, dead, motion, gt, runs in vs:
            codes0 = possession.label(v, dead, possession.PARAMS, motion)
            rows.append((v, dead, gt, runs, owner.features(v, codes0, motion), codes0))
        prepared.append(rows)

    def spell(row, ph, lam):
        v, dead, gt, runs, _, _ = row
        pred = owner.label(v, dead, ph, lam)
        pred[gt == harness.X] = harness.X
        return (pm.spell_scores(gt, pred, runs)["mean_f1"],
                pm.macro_f1(pm.scored_confusion(pm.confusion(gt, pred))))

    def train(ms):
        Xs, ys = [], []
        for m in ms:
            for v, dead, gt, runs, X, _ in prepared[m]:
                mask = ~dead & np.isin(gt, (0, 1))
                mask[np.arange(len(mask)) % (v["fps"] // SHAPE_HZ) != 0] = False
                Xs.append(X[mask])
                ys.append((gt[mask] == possession.HOME).astype(float))
        model = owner.fit(np.concatenate(Xs), np.concatenate(ys))
        probs = {m: [owner.prob(model, row[4]) for row in prepared[m]] for m in ms}
        lam = max(OWNER_GRID, key=lambda c: np.mean([spell(row, ph, c)[0] for m in ms
                                                     for row, ph in zip(prepared[m], probs[m])]))
        return model, lam

    every = list(range(len(match_ids)))
    held, rules, acc, rules_acc = [], [], [], []
    for mi in every:
        model, lam = train([j for j in every if j != mi])
        s = []
        for row in prepared[mi]:
            v, dead, gt, runs, X, codes0 = row
            ph = owner.prob(model, X)
            s.append(spell(row, ph, lam))
            mask = ~dead & np.isin(gt, (0, 1))
            acc.append(((ph[mask] >= 0.5) == (gt[mask] == possession.HOME)).mean())
            rules_acc.append((codes0[mask] == gt[mask]).mean())
            r = codes0.copy()
            r[gt == harness.X] = harness.X
            rules.append((mi, pm.spell_scores(gt, r, runs)["mean_f1"]))
        held.append((match_ids[mi], lam, np.mean(s, axis=0)))
        print("  fold %s done: switch cost %g, spellF1 %.3f" % (match_ids[mi], lam, held[-1][2][0]),
              flush=True)
    model, lam = train(every)
    r = np.array(rules)
    return {"held": held, "model": model, "lam": lam, "acc": float(np.mean(acc)),
            "rules_acc": float(np.mean(rules_acc)),
            "rules": np.array([r[r[:, 0] == m, 1].mean() for m in every])}


def owner_report(r=None):
    from . import owner
    r = owner_run() if r is None else r
    lines = ["owner model over all live play (owner.py), IDSSE end to end, nested leave-one-match-out",
             "  match    spellF1  macroF1  (rules 1-4 alone)  switch cost"]
    for (mid, lam, s), b in zip(r["held"], r["rules"]):
        lines.append("  {}   {:.3f}    {:.3f}    ({:.3f})            {:g}".format(mid, s[0], s[1], b, lam))
    h = np.array([s for _, _, s in r["held"]])
    lines += ["  mean     {:.3f}    {:.3f}    ({:.3f})".format(h[:, 0].mean(), h[:, 1].mean(),
                                                            r["rules"].mean()),
              "  owner accuracy on live frames, held out: model {:.1%}, rules 1-4 {:.1%}".format(
                  r["acc"], r["rules_acc"]),
              "", "fitted on all seven: switch cost %g" % r["lam"],
              "  standardised weights (towards home having the ball):"]
    for name, x in sorted(zip(("bias",) + owner.NAMES, r["model"][2]), key=lambda t: -abs(t[1])):
        lines.append("    {:<28} {:+.2f}".format(name, x))
    return "\n".join(lines)


# ---- learned change detector (changes.py) -------------------------------------------------------
CHANGE_GRID = (0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.8)   # THRESH


def changes_run(match_ids=external.MATCHES):
    """Nested leave-one-match-out: the change model is fitted, and its threshold chosen, on six
    matches and scored once on the seventh, end to end (our dead ball, ball hidden), against
    rules 1-4 on the same frames. Neither side uses the shape model (rule 6), so nothing fitted
    on the held-out match enters."""
    from . import changes
    ev = Evaluator(match_ids)
    prepared = []                         # per match: (v, dead, gt, runs, frames, teams, X, Xtr, ytr)
    for vs in ev.items:
        rows = []
        for v, dead, motion, gt, runs in vs:
            fr, tm, X = changes.candidates(v, motion)
            Xtr, ytr = changes.training_rows(fr, tm, X, gt, dead, v["fps"])
            rules = possession.label(v, dead, possession.PARAMS, motion)
            rules[gt == harness.X] = harness.X
            rows.append((v, dead, gt, runs, fr, tm, X, Xtr, ytr,
                         pm.spell_scores(gt, rules, runs)["mean_f1"],
                         pm.macro_f1(pm.scored_confusion(pm.confusion(gt, rules)))))
        prepared.append(rows)

    def spell(row, p, thresh):
        v, dead, gt, runs, fr, tm = row[:6]
        pred = changes.label(v, dead, fr, tm, p, thresh)
        pred[gt == harness.X] = harness.X
        s = pm.spell_scores(gt, pred, runs)
        return s["mean_f1"], pm.macro_f1(pm.scored_confusion(pm.confusion(gt, pred))), \
            s["n_pred"] / max(s["n_gt"], 1)

    def train(ms):
        model = changes.fit(np.concatenate([row[7] for m in ms for row in prepared[m]]),
                            np.concatenate([row[8] for m in ms for row in prepared[m]]))
        probs = {m: [changes.prob(model, row[6]) for row in prepared[m]] for m in ms}
        thresh = max(CHANGE_GRID, key=lambda c: np.mean([spell(row, p, c)[0] for m in ms
                                                         for row, p in zip(prepared[m], probs[m])]))
        return model, thresh

    every = list(range(len(match_ids)))
    held, base = [], []
    for mi in every:
        model, thresh = train([j for j in every if j != mi])
        s = [spell(row, changes.prob(model, row[6]), thresh) for row in prepared[mi]]
        held.append((match_ids[mi], thresh, np.mean(s, axis=0)))
        base.append(np.mean([(row[9], row[10]) for row in prepared[mi]], axis=0))
        print("  fold %s: threshold %g, spellF1 %.3f (rules %.3f)" % (
            match_ids[mi], thresh, held[-1][2][0], base[-1][0]), flush=True)
    model, thresh = train(every)
    return {"held": held, "base": np.array(base), "model": model, "thresh": thresh}


def changes_report(r=None):
    from . import changes
    r = changes_run() if r is None else r
    lines = ["learned change detector (changes.py), IDSSE end to end, nested leave-one-match-out",
             "  match    spellF1 (rules 1-4)   macroF1 (rules 1-4)   spells/true   threshold"]
    for (mid, th, s), b in zip(r["held"], r["base"]):
        lines.append("  {}   {:.3f} ({:.3f})         {:.3f} ({:.3f})         {:.2f}          {:g}".format(
            mid, s[0], b[0], s[1], b[1], s[2], th))
    h = np.array([s for _, _, s in r["held"]])
    d = h[:, 0] - r["base"][:, 0]
    lines += ["  mean     {:.3f} ({:.3f})         {:.3f} ({:.3f})         {:.2f}".format(
        h[:, 0].mean(), r["base"][:, 0].mean(), h[:, 1].mean(), r["base"][:, 1].mean(), h[:, 2].mean()),
        "  spellF1 against rules 1-4: {:+.3f}, better on {}/7".format(d.mean(), (d > 0).sum()),
        "", "fitted on all seven: threshold %g" % r["thresh"],
        "  standardised weights (towards a real change):"]
    for name, x in sorted(zip(("bias",) + changes.NAMES, r["model"][2]), key=lambda t: -abs(t[1])):
        lines.append("    {:<26} {:+.2f}".format(name, x))
    return "\n".join(lines)


def changes_shape_run(match_ids=external.MATCHES):
    """The change detector with rule 6 on top, everything fitted inside each fold: the change
    model and threshold, then the shape model (on the detector's own first pass) and its switch
    rule from SHAPE_GRID, all on six matches; scored once on the seventh."""
    import itertools
    from . import changes, shape
    ev = Evaluator(match_ids)
    prepared = []
    for vs in ev.items:
        rows = []
        for v, dead, motion, gt, runs in vs:
            fr, tm, X = changes.candidates(v, motion)
            Xtr, ytr = changes.training_rows(fr, tm, X, gt, dead, v["fps"])
            rows.append((v, dead, gt, runs, fr, tm, X, Xtr, ytr))
        prepared.append(rows)
    combos = [dict(zip(SHAPE_GRID, c)) for c in itertools.product(*SHAPE_GRID.values())]

    def score(gt, pred, runs):
        pred = pred.copy()
        pred[gt == harness.X] = harness.X
        return (pm.spell_scores(gt, pred, runs)["mean_f1"],
                pm.macro_f1(pm.scored_confusion(pm.confusion(gt, pred))))

    def train(ms):
        model = changes.fit(np.concatenate([r[7] for m in ms for r in prepared[m]]),
                            np.concatenate([r[8] for m in ms for r in prepared[m]]))
        p = {m: [changes.prob(model, r[6]) for r in prepared[m]] for m in ms}
        thresh = max(CHANGE_GRID, key=lambda c: np.mean([
            score(r[2], changes.label(r[0], r[1], r[4], r[5], pp, c), r[3])[0]
            for m in ms for r, pp in zip(prepared[m], p[m])]))
        first = {m: [changes.label(r[0], r[1], r[4], r[5], pp, thresh)
                     for r, pp in zip(prepared[m], p[m])] for m in ms}
        Xs, ys = [], []
        for m in ms:
            for r, c0 in zip(prepared[m], first[m]):
                v, dead, gt = r[0], r[1], r[2]
                X = shape.inputs(v, c0)
                mask = ~v["ball_seen"] & ~dead & np.isin(gt, (0, 1))
                mask[np.arange(len(mask)) % (v["fps"] // SHAPE_HZ) != 0] = False
                Xs.append(X[mask])
                ys.append((gt[mask] == possession.HOME).astype(float))
        smodel = shape.fit(np.concatenate(Xs), np.concatenate(ys))
        ph = {m: [shape.prob(smodel, shape.inputs(r[0], c0)) for r, c0 in zip(prepared[m], first[m])]
              for m in ms}
        q = max(combos, key=lambda q: np.mean([
            score(r[2], changes.label(r[0], r[1], r[4], r[5], pp, thresh, p_home=h, q=q), r[3])[0]
            for m in ms for r, pp, h in zip(prepared[m], p[m], ph[m])]))
        return model, thresh, smodel, q

    every = list(range(len(match_ids)))
    held = []
    for mi in every:
        model, thresh, smodel, q = train([j for j in every if j != mi])
        s = []
        for r in prepared[mi]:
            pp = changes.prob(model, r[6])
            c0 = changes.label(r[0], r[1], r[4], r[5], pp, thresh)
            h = shape.prob(smodel, shape.inputs(r[0], c0))
            s.append(score(r[2], changes.label(r[0], r[1], r[4], r[5], pp, thresh, p_home=h, q=q), r[3]))
        held.append((match_ids[mi], thresh, q, np.mean(s, axis=0)))
        print("  fold %s: threshold %g, %s, spellF1 %.3f" % (
            match_ids[mi], thresh, " ".join("%s=%g" % kv for kv in q.items()), held[-1][3][0]), flush=True)
    model, thresh, smodel, q = train(every)
    return {"held": held, "model": model, "thresh": thresh, "shape_model": smodel, "q": q}


def changes_shape_report(r=None):
    from . import changes, shape
    r = changes_shape_run() if r is None else r
    lines = ["change detector (changes.py) with rule 6, IDSSE end to end, nested leave-one-match-out",
             "  match    spellF1  macroF1   threshold   rule 6"]
    for mid, th, q, s in r["held"]:
        lines.append("  {}   {:.3f}    {:.3f}     {:<9g}   {}".format(
            mid, s[0], s[1], th, " ".join("%s=%g" % kv for kv in q.items())))
    h = np.array([s for _, _, _, s in r["held"]])
    lines += ["  mean     {:.3f}    {:.3f}".format(h[:, 0].mean(), h[:, 1].mean()), "",
              "fitted on all seven: THRESH %g, %s" % (r["thresh"], " ".join("%s=%g" % kv for kv in r["q"].items())),
              "  change model, standardised weights (towards a real change):"]
    for name, x in sorted(zip(("bias",) + changes.NAMES, r["model"][2]), key=lambda t: -abs(t[1])):
        lines.append("    {:<26} {:+.2f}".format(name, x))
    for label, model in (("changes.MODEL", r["model"]), ("shape.MODEL", r["shape_model"])):
        lines.append("\n%s (mean, sd, weights), as stored:" % label)
        for part in model:
            lines.append("    np.array([{}]),".format(", ".join("%g" % float("%.6g" % x) for x in part)))
    return "\n".join(lines)
