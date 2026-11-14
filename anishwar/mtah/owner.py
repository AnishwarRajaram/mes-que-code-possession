"""Who has the ball in live play, read frame by frame and decoded as a sequence.

The touch rules (possession.py, rules 1-4) need one clear touch to change hands. On other matches
(IDSSE) most of their remaining error is with the ball in view: a defender beside a pass trips a
touch and the change stands for seconds, because the team really on the ball makes no touch the
thresholds catch; or a real change trips none. This module reads the owner from evidence that
accumulates instead:

  ball proximity     which team is nearer the ball now, over the last and next second, and over
                     three seconds (clipped at PROX_CLIP_M; zero when the ball is not tracked)
  ball speed         a fast ball says less about who is near it (speed, and speed x proximity)
  touch rules        what rules 1-4 say (+1 home, -1 away), as evidence, not as a switch
  team shape         shape.team_features, which is all there is when the ball is not tracked

A logistic model (challenger.fit) turns them into a probability that home has the ball, trained on
IDSSE live frames with the DFL owner as the label. A two-state hidden Markov model then decodes
each live stretch: every frame adds its log-odds (per second), every change of hands costs
SWITCH_COST. The control filter (possession.control_filter) is applied afterwards as before.
"""

import numpy as np
import pandas as pd

from . import challenger, deadball, possession

PROX_CLIP_M = 10.0     # beyond this, being "nearer the ball" carries no information (fixed)
WINDOWS_S = (1.0, 3.0)
NAMES = ("rules say", "ball tracked", "proximity now", "proximity last 1 s", "proximity next 1 s",
         "proximity +-3 s", "ball speed", "ball speed x proximity",
         "forward velocity", "forward position", "depth", "width", "in the other half")


def features(cues, codes0, motion):
    """(n, len(NAMES)) inputs; positive values point to home. codes0: rules 1-4 on this match."""
    fps = cues["fps"]
    seen = cues["ball_seen"].astype(float)
    dh = np.nan_to_num(cues["near_home"], nan=PROX_CLIP_M)
    da = np.nan_to_num(cues["near_away"], nan=PROX_CLIP_M)
    prox = (np.clip(da, 0, PROX_CLIP_M) - np.clip(dh, 0, PROX_CLIP_M)) * seen
    speed = np.nan_to_num(motion[0], nan=0.0) * seen
    s = pd.Series(prox)
    k1, k3 = int(WINDOWS_S[0] * fps), int(WINDOWS_S[1] * fps)
    past = s.rolling(k1, min_periods=1).mean().values
    nxt = s[::-1].rolling(k1, min_periods=1).mean()[::-1].values
    wide = s.rolling(2 * k3, center=True, min_periods=1).mean().values
    rules = np.where(codes0 == possession.HOME, 1.0, np.where(codes0 == possession.AWAY, -1.0, 0.0))
    cols = [rules, seen, prox, past, nxt, wide, speed, speed * prox / PROX_CLIP_M]
    return np.column_stack(cols + [cues["shape"][:, j] for j in range(5)]).astype(np.float32)


def fit(X, y):
    return challenger.fit(X, y)


def prob(model, X):
    return challenger.prob(model, X)


def viterbi(logit, starts, lengths, lam, init=None, end=None):
    """Most likely home(0)/away(1) path through each stretch [starts[k], starts[k]+lengths[k]).
    logit: per-frame log-odds for home, already per second. lam: cost of a change of hands.
    init / end: per stretch, the state fixed before the first frame / at the last frame, or -1
    (free). Same recursion as possession.viterbi_override: d[t] = e[t] + clip(d[t-1], -lam, lam).
    Returns a list of int8 arrays, one per stretch."""
    G = len(starts)
    if G == 0:
        return []
    starts, lengths = np.asarray(starts), np.asarray(lengths)
    init = np.full(G, -1) if init is None else np.asarray(init)
    end = np.full(G, -1) if end is None else np.asarray(end)
    T = int(lengths.max())
    t_idx = np.arange(T)
    frames = np.minimum(starts[:, None] + t_idx[None, :], len(logit) - 1)
    e = np.where(t_idx[None, :] < lengths[:, None], logit[frames], 0.0)
    d = np.empty((G, T))
    prev = np.where(init == 0, np.inf, np.where(init == 1, -np.inf, 0.0))
    for t in range(T):
        d[:, t] = e[:, t] + np.clip(prev, -lam, lam)
        prev = d[:, t]
    last = d[np.arange(G), lengths - 1]
    state = np.where(end >= 0, end, np.where(last >= 0, 0, 1))
    path = np.empty((G, T), np.int8)
    for t in range(T - 1, 0, -1):
        path[:, t] = state
        dp = d[:, t - 1]
        before = np.where(state == 0, np.where(dp >= -lam, 0, 1), np.where(dp <= lam, 1, 0))
        state = np.where(t < lengths, before, state)
    path[:, 0] = state
    return [path[k, :lengths[k]] for k in range(G)]


def label(cues, dead, p_home, lam):
    """Codes 0 home / 1 away / 2 dead: every live stretch decoded from p_home, then the control
    filter. The halves are decoded apart."""
    fps, n = cues["fps"], len(dead)
    p = np.clip(p_home, 1e-4, 1 - 1e-4)
    logit = np.log(p / (1 - p)) / fps
    codes = np.full(n, possession.DEAD, np.int8)
    starts, lengths = [], []
    for sg in np.unique(cues["segment"]):
        idx = np.flatnonzero(cues["segment"] == sg)
        a, b = deadball.runs(~dead[idx])
        starts += list(idx[0] + a)
        lengths += list(b - a)
    for s, ln, path in zip(starts, lengths, viterbi(logit, starts, lengths, lam)):
        codes[s:s + ln] = path
    return possession.control_filter(codes, cues["segment"], fps)
