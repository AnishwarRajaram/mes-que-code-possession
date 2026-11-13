"""Score engine: dead-ball evidence as named points, the weights learned on other matches.

Each frame gets a score = bias + sum of points x evidence. Every evidence term is a plain,
bounded quantity in [0, 1] that an analyst can read, and every term has a fixed direction set by
football: slow players are evidence of a dead ball, fast players of live play. The weights are the
only thing learned: a logistic regression on IDSSE (DFL ball status) with each weight held to its
sign, so the model can down-weight a cue but never turn it backwards. A weight is "points per
full unit of evidence", in log-odds.

The kick-off shape gets a stronger prior towards zero (SHAPE_PRIOR): it is a rare event and its
per-frame separation is weak, so it should only carry weight the data clearly asks for.

The ball being missing is deliberately not a term (see challenger.py for why).

Judged by the same nested leave-one-match-out as the rules (calibrate.py), with the same decision
step as the challenger (smoothing, threshold, minimum run lengths chosen on the training matches).
Aggregates per term: its points, its mean contribution on dead and on live frames, and how much
held-out score is lost when it is removed.
"""

import numpy as np
import pandas as pd

from . import calibrate, challenger, deadball, external

SLOW_SPAN = 1.0           # slow = clip(1.5 - ratio, 0, 1): 0 at 1.5x the usual pace, 1 at 0.5x
OUT_DECAY_S = (3.0, 12.0)  # memory of a ball-out: quick restart, and a slower one
INFERRED_NEAR_M = 3.0      # used when the rules leave rule 4 off: last seen within 3 m of a line
INFERRED_GAP_S = 2.0       # ... then unseen for 2 s
KO_SHARE = 0.85            # kick-off shape: each team this much in its own half, a player on the spot
L2_PER_ROW = 1e-5
SHAPE_PRIOR = 100.0        # kick-off shape penalised this many times harder
TRAIN_HZ = 5
NEWTON_STEPS = 30

# name, sign (+1 dead evidence, -1 live evidence)
TERMS = (
    ("players slow now", +1),
    ("players slow, last 4 s", +1),
    ("players slow, next 4 s", +1),
    ("players slow, +-12 s", +1),
    ("players fast now", -1),
    ("ball seen out, recent (3 s)", +1),
    ("ball seen out, recent (12 s)", +1),
    ("ball vanished at a line, recent (12 s)", +1),
    ("kick-off shape", +1),
)
NAMES = [t[0] for t in TERMS]
SIGNS = np.array([t[1] for t in TERMS], float)


def _decay(events, n, fps, tau_s):
    """exp(-time since the last event / tau), 0 before any event."""
    last = np.full(n, -np.inf)
    last[events] = events
    last = np.maximum.accumulate(last)
    dt = (np.arange(n) - last) / fps
    return np.where(np.isfinite(dt), np.exp(-dt / tau_s), 0.0)


def design(c):
    fps, n = c["fps"], len(c["ratio"])
    r = np.nan_to_num(c["ratio"], nan=1.0)
    slow = pd.Series(np.clip(1.5 - r, 0, 1) / SLOW_SPAN)
    k4, k12 = int(4 * fps), int(12 * fps)
    out = np.nan_to_num(c["beyond"], nan=-1.0) > 0
    seen_out, _ = deadball.runs(out)
    p = dict(deadball.PARAMS)
    if p["OUT_NEAR_M"] <= 0:
        p.update(OUT_NEAR_M=INFERRED_NEAR_M, OUT_GAP_S=INFERRED_GAP_S)
    vanished = deadball.inferred_outs(c, p)
    ko = pd.Series(((c["ko_side"] >= KO_SHARE) & c["ko_spot"]).astype(float))
    cols = [
        slow.values,
        slow.rolling(k4, min_periods=1).mean().values,
        slow[::-1].rolling(k4, min_periods=1).mean()[::-1].values,
        slow.rolling(2 * k12, center=True, min_periods=1).mean().values,
        np.clip(r - 1.0, 0, 1),
        _decay(seen_out, n, fps, OUT_DECAY_S[0]),
        _decay(seen_out, n, fps, OUT_DECAY_S[1]),
        _decay(vanished, n, fps, 12.0),
        ko.rolling(fps, center=True, min_periods=1).mean().values,
    ]
    return np.column_stack(cols).astype(np.float32)


def fit(X, y, active=None):
    """Sign-held logistic regression: Newton on the active terms, any term that turns against
    its sign is dropped and the fit repeated. Returns weights (bias first), zero for dropped."""
    k = X.shape[1]
    active = np.ones(k, bool) if active is None else active.copy()
    pen_all = np.full(k, L2_PER_ROW * len(X))
    pen_all[NAMES.index("kick-off shape")] *= SHAPE_PRIOR
    while True:
        idx = np.flatnonzero(active)
        Z = np.column_stack([np.ones(len(X)), X[:, idx]])
        pen = np.concatenate(([0.0], pen_all[idx]))
        w = np.zeros(Z.shape[1])
        for _ in range(NEWTON_STEPS):
            q = 1 / (1 + np.exp(-np.clip(Z @ w, -30, 30)))
            g = Z.T @ (q - y) + pen * w
            H = (Z * (q * (1 - q))[:, None]).T @ Z + np.diag(pen)
            step = np.linalg.solve(H, g)
            w -= step
            if np.abs(step).max() < 1e-6:
                break
        wrong = w[1:] * SIGNS[idx] < 0
        if not wrong.any():
            full = np.zeros(k + 1)
            full[0] = w[0]
            full[1 + idx] = w[1:]
            return full
        worst = idx[np.argmin(w[1:] * SIGNS[idx])]
        active[worst] = False


def prob(w, X):
    return 1 / (1 + np.exp(-np.clip(w[0] + X @ w[1:], -30, 30)))


def _train(ev, data, ms, active, posts):
    step = ev.variants[0][0]["fps"] // TRAIN_HZ
    X = np.concatenate([x[::step] for m in ms for x in data[m]])
    y = np.concatenate([c["dead"][::step] for m in ms for c in ev.variants[m]]).astype(float)
    w = fit(X, y, active)
    probs = {m: [prob(w, x) for x in data[m]] for m in ms}
    best = max(posts, key=lambda q: np.mean([
        challenger._score(challenger.decide(p, c["fps"], q), c)[:2].mean()
        for m in ms for p, c in zip(probs[m], ev.variants[m])]))
    return w, best


def _nested(ev, data, active, posts):
    every = list(range(len(ev.variants)))
    held = []
    for mi in every:
        w, post = _train(ev, data, [j for j in every if j != mi], active, posts)
        held.append(np.mean([challenger._score(challenger.decide(prob(w, x), c["fps"], post), c)
                             for x, c in zip(data[mi], ev.variants[mi])], axis=0))
    return np.array(held)


def run(match_ids=external.MATCHES, use_dropout=True, ablation=True):
    import itertools
    ev = calibrate.Evaluator(match_ids, use_dropout)
    data = [[design(c) for c in vs] for vs in ev.variants]
    posts = [dict(zip(challenger.POST_GRID, v))
             for v in itertools.product(*challenger.POST_GRID.values())]
    k = len(NAMES)
    held = _nested(ev, data, np.ones(k, bool), posts)
    w, post = _train(ev, data, list(range(len(match_ids))), np.ones(k, bool), posts)

    X = np.concatenate([x for xs in data for x in xs])
    dead = np.concatenate([c["dead"] for vs in ev.variants for c in vs])
    contrib = X * w[1:]
    per_term = [(NAMES[j], w[1 + j], contrib[dead, j].mean(), contrib[~dead, j].mean())
                for j in range(k)]
    drops = {}
    if ablation:
        base = held[:, :2].mean()
        for j in range(k):
            active = np.ones(k, bool)
            active[j] = False
            drops[NAMES[j]] = base - _nested(ev, data, active, posts)[:, :2].mean()
    return {"ids": match_ids, "held": held, "weights": w, "post": post, "per_term": per_term,
            "drops": drops}


def report(use_dropout=True, ablation=True):
    r = run(use_dropout=use_dropout, ablation=ablation)
    h = r["held"]
    lines = ["score engine, IDSSE, ball {}".format(
        "hidden like ours" if use_dropout else "fully tracked"),
        "", "nested leave-one-match-out:", "  match    stopF1  frameF1  dead%  stops/90"]
    for mid, s in zip(r["ids"], h):
        lines.append("  {}   {:.3f}   {:.3f}   {:4.1f}  {:5.0f}".format(mid, s[0], s[1],
                                                                       100 * s[2], s[3]))
    lines += ["  mean     {:.3f}   {:.3f}   {:4.1f}  {:5.0f}".format(
        h[:, 0].mean(), h[:, 1].mean(), 100 * h[:, 2].mean(), h[:, 3].mean()),
        "  sd       {:.3f}   {:.3f}".format(h[:, 0].std(), h[:, 1].std()),
        "", "fitted on all seven: bias {:+.2f}; decision {}".format(
            r["weights"][0], " ".join("%s=%g" % kv for kv in r["post"].items())),
        "  {:<40} {:>7} {:>11} {:>11} {:>14}".format("evidence", "points", "mean dead",
                                                     "mean live", "held-out loss")]
    for name, pts, d, l in r["per_term"]:
        drop = r["drops"].get(name)
        lines.append("  {:<40} {:+7.2f} {:+11.2f} {:+11.2f} {:>14}".format(
            name, pts, d, l, "" if drop is None else "{:+.4f}".format(drop)))
    lines.append("  (held-out loss: drop in mean of stoppage and frame F1 when the term is removed)")
    return "\n".join(lines)
