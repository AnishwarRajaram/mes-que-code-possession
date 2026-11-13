"""A small learned challenger to the dead-ball rules: logistic regression on the same cues.

It sees only what the rules see (players' speed against the local norm, the ball past a line),
summarised over windows before and after each frame, so it can learn shapes the rules write by
hand: the slow-down after a whistle, the speed-up before a restart. It is fitted on other matches
(IDSSE, DFL ball status) with the ball hidden the way our tracking loses it, and judged by the
same nested leave-one-match-out as the rules (calibrate.py). It replaces the rules only if it
beats them there; the provided game never chooses.

Deterministic: standardised features, L2-penalised Newton steps (IRLS) in numpy, no random
initialisation. Ball-missing itself is deliberately NOT a feature: on IDSSE the gaps are placed
by our profile, so the model would only learn our placement rule back.

Decision: probability smoothed over SMOOTH_S, cut at a threshold, then dead runs shorter than
MIN_DEAD_S dropped and live gaps shorter than MIN_LIVE_S filled. The threshold and both lengths
are chosen on the training matches of each fold, from POST_GRID.
"""

import itertools

import numpy as np
import pandas as pd

from . import calibrate, deadball, external

WINDOWS_S = (1, 3, 6, 12)        # look before and after: a touch, a restart, a set piece, a VAR-free pause
OUT_MEMORY_S = (3, 10)           # the ball went out recently
TRAIN_HZ = 5                     # frames used for fitting: 5 per second is enough for a smooth cue
L2 = 1.0
NEWTON_STEPS = 25
SMOOTH_S = 1.0
POST_GRID = {"threshold": (0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65),
             "MIN_DEAD_S": (1.0, 2.0, 4.0),
             "MIN_LIVE_S": (1.0, 2.0, 4.0)}


def design(c):
    """(n, k) feature matrix and names for one match's cues."""
    fps = c["fps"]
    r = pd.Series(np.clip(np.nan_to_num(c["ratio"], nan=1.0), 0, 3))
    out = pd.Series((np.nan_to_num(c["beyond"], nan=-1.0) > 0).astype(float))
    cols, names = [r.values, np.log(r.values + 0.05)], ["ratio", "log ratio"]
    for w in WINDOWS_S:
        k = int(w * fps)
        past = r.rolling(k, min_periods=1).mean()
        future = r[::-1].rolling(k, min_periods=1).mean()[::-1]
        cols += [past.values, future.values, r.rolling(2 * k, center=True, min_periods=1).min().values]
        names += ["ratio past %ds" % w, "ratio next %ds" % w, "ratio min +-%ds" % w]
    for w in OUT_MEMORY_S:
        cols.append(out.rolling(int(w * fps), min_periods=1).max().values)
        names.append("ball out in last %ds" % w)
    return np.column_stack(cols), names


def fit(X, y):
    """L2 logistic regression by Newton steps. Returns (mean, sd, weights incl. bias)."""
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = np.column_stack([np.ones(len(X)), (X - mu) / sd])
    w = np.zeros(Z.shape[1])
    pen = np.full(Z.shape[1], L2)
    pen[0] = 0.0
    for _ in range(NEWTON_STEPS):
        p = 1 / (1 + np.exp(-np.clip(Z @ w, -30, 30)))
        g = Z.T @ (p - y) + pen * w
        H = (Z * (p * (1 - p))[:, None]).T @ Z + np.diag(pen)
        step = np.linalg.solve(H, g)
        w -= step
        if np.abs(step).max() < 1e-6:
            break
    return mu, sd, w


def prob(model, X):
    mu, sd, w = model
    return 1 / (1 + np.exp(-np.clip(w[0] + ((X - mu) / sd) @ w[1:], -30, 30)))


def decide(p, fps, post):
    p = pd.Series(p).rolling(int(SMOOTH_S * fps), center=True, min_periods=1).mean().values
    dead = p >= post["threshold"]
    a, b = deadball.runs(~dead)
    for s, e in zip(a, b):                       # fill short live gaps
        if 0 < s and e < len(dead) and e - s < post["MIN_LIVE_S"] * fps:
            dead[s:e] = True
    a, b = deadball.runs(dead)
    for s, e in zip(a, b):                       # drop short dead runs
        if e - s < post["MIN_DEAD_S"] * fps:
            dead[s:e] = False
    return dead


def _score(dead, c):
    s = deadball.score(dead, c["dead"], c["segment"], c["fps"])
    return np.array([s["f1_stop"], s["f1_frame"], s["dead_share"], s["stops_per90"]])


def run(match_ids=external.MATCHES, use_dropout=True):
    ev = calibrate.Evaluator(match_ids, use_dropout)          # same ball-hidden variants
    data = [[design(c)[0] for c in vs] for vs in ev.variants]
    names = design(ev.variants[0][0])[1]
    posts = [dict(zip(POST_GRID, v)) for v in itertools.product(*POST_GRID.values())]

    def train(ms):
        step = ev.variants[0][0]["fps"] // TRAIN_HZ
        X = np.concatenate([x[::step] for m in ms for x in data[m]])
        y = np.concatenate([c["dead"][::step] for m in ms for c in ev.variants[m]]).astype(float)
        model = fit(X, y)
        probs = {m: [prob(model, x) for x in data[m]] for m in ms}
        best = max(posts, key=lambda q: np.mean([
            _score(decide(p, c["fps"], q), c)[:2].mean()
            for m in ms for p, c in zip(probs[m], ev.variants[m])]))
        return model, best

    every = list(range(len(match_ids)))
    held = []
    for mi in every:
        model, post = train([j for j in every if j != mi])
        s = np.mean([_score(decide(prob(model, x), c["fps"], post), c)
                     for x, c in zip(data[mi], ev.variants[mi])], axis=0)
        held.append((match_ids[mi], post, s))
    model, post = train(every)
    return {"held_out": held, "post": post, "weights": dict(zip(["bias"] + names, model[2])),
            "dropout": use_dropout}


def report(use_dropout=True):
    r = run(use_dropout=use_dropout)
    lines = ["logistic challenger, IDSSE, ball {}".format(
        "hidden like ours" if use_dropout else "fully tracked"),
        "", "nested leave-one-match-out:", "  match    stopF1  frameF1  dead%  stops/90   post"]
    for mid, q, s in r["held_out"]:
        lines.append("  {}   {:.3f}   {:.3f}   {:4.1f}  {:5.0f}     {}".format(
            mid, s[0], s[1], 100 * s[2], s[3], " ".join("%s=%g" % kv for kv in q.items())))
    h = np.array([s for _, _, s in r["held_out"]])
    lines += ["  mean     {:.3f}   {:.3f}   {:4.1f}  {:5.0f}".format(
        h[:, 0].mean(), h[:, 1].mean(), 100 * h[:, 2].mean(), h[:, 3].mean()),
        "  sd       {:.3f}   {:.3f}".format(h[:, 0].std(), h[:, 1].std()),
        "", "fitted on all seven, standardised weights (largest first):"]
    for k, v in sorted(r["weights"].items(), key=lambda kv: -abs(kv[1])):
        lines.append("  {:<22} {:+.2f}".format(k, v))
    return "\n".join(lines)
