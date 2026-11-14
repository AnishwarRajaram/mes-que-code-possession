"""A learned change detector: which near-ball moments are real changes of possession.

spellF1 counts every possession spell the same, however short, so a missed change costs as much
as a long possession read wrong. The touch rules (possession.py, rule 1) need one large turn or
speed change; on other matches (IDSSE) they miss many real changes and accept a defender beside a
pass. Here the net is cast wide and a model judges each catch:

  1. Candidates: the ball tracked, a player within CAND_R of it, and the ball turning by CAND_TURN
     or changing speed by CAND_JUMP -- loose on purpose, so almost every real change is caught.
     A run of consecutive candidate frames of one team is one candidate, at its strongest frame.
  2. Features of each candidate, from the side of the team that would take the ball: turn, speed
     change, ball speed before and after, that team's and the other team's distance to the ball,
     who stays nearer the ball over the next 1 and 2 s and who was nearer over the last 1 s, how
     much of the next second the ball is unseen, how the two teams' forward velocity changes, and
     whether rule 1 would also call it a touch.
  3. A logistic model (challenger.fit) gives the probability that the candidate is a real change;
     it is trained on IDSSE candidates against the team holding the ball, labelled by whether the
     DFL owner (through the control filter) changes to that team within LABEL_S.
  4. Decoding: through each live stretch in order, a candidate of the other team with probability
     at least THRESH changes hands. The first candidate owns the restart (rule 3); then rules 4-6
     and the control filter as before.
The candidate constants are fixed in advance (loose by design) and never tuned; THRESH is
calibrated end to end by nested leave-one-match-out.
"""

import numpy as np
import pandas as pd

from . import challenger, deadball, possession

CAND_R = 2.5          # m: a player this near the ball can have touched it
CAND_TURN = 5.0       # degrees
CAND_JUMP = 2.0       # m/s
PROX_CLIP_M = 10.0
LABEL_S = 0.4         # a real change starts within this of the candidate
NAMES = ("turn", "speed change", "ball speed before", "ball speed after", "taker distance",
         "other distance", "taker nearer next 1 s", "taker nearer next 2 s",
         "taker nearer last 1 s", "ball unseen next 1 s", "forward velocity change",
         "rule 1 touch")


def candidates(cues, motion, p_rules=possession.PARAMS):
    """(frames, teams, X): one row per candidate, features from the taking team's side."""
    fps = cues["fps"]
    speed, turn, jump = motion
    seen = cues["ball_seen"]
    dh = np.nan_to_num(cues["near_home"], nan=np.inf)
    da = np.nan_to_num(cues["near_away"], nan=np.inf)
    with np.errstate(invalid="ignore"):
        act = (np.nan_to_num(turn) >= CAND_TURN) | (np.abs(np.nan_to_num(jump)) >= CAND_JUMP)
    near_team = np.where(dh <= da, possession.HOME, possession.AWAY)
    cand = seen & act & (np.minimum(dh, da) <= CAND_R)
    strength = np.nan_to_num(turn) / 30.0 + np.abs(np.nan_to_num(jump)) / 5.0
    # one candidate per run of consecutive candidate frames of the same team
    key = np.where(cand, near_team, -1)
    frames = []
    a, b = deadball.runs(cand)
    for s, e in zip(a, b):
        cut = np.flatnonzero(np.diff(key[s:e]) != 0) + 1
        for x, y in zip(np.r_[0, cut], np.r_[cut, e - s]):
            frames.append(s + x + int(np.argmax(strength[s + x:s + y])))
    frames = np.array(frames, int)
    teams = near_team[frames] if len(frames) else np.zeros(0, int)

    # per-frame series from the home side; flipped for away takers
    prox = (np.clip(np.where(np.isinf(da), PROX_CLIP_M, da), 0, PROX_CLIP_M)
            - np.clip(np.where(np.isinf(dh), PROX_CLIP_M, dh), 0, PROX_CLIP_M)) * seen
    s = pd.Series(prox)
    k1, k2 = fps, 2 * fps
    nxt1 = s[::-1].rolling(k1, min_periods=1).mean()[::-1].shift(-1).fillna(0).values
    nxt2 = s[::-1].rolling(k2, min_periods=1).mean()[::-1].shift(-1).fillna(0).values
    last1 = s.rolling(k1, min_periods=1).mean().shift(1).fillna(0).values
    unseen1 = pd.Series((~seen).astype(float))[::-1].rolling(k1, min_periods=1).mean()[::-1].values
    fv = pd.Series(cues["shape"][:, 0].astype(float))
    fv_change = (fv[::-1].rolling(k2, min_periods=1).mean()[::-1].values
                 - fv.rolling(k2, min_periods=1).mean().values)
    rule1 = np.zeros(len(seen), bool)
    tf, _ = possession.touches(cues, p_rules, motion)
    rule1[tf] = True

    f = frames
    sign = np.where(teams == possession.HOME, 1.0, -1.0)
    d_take = np.where(teams == possession.HOME, dh[f], da[f])
    d_other = np.where(teams == possession.HOME, da[f], dh[f])
    sp, jp = np.nan_to_num(speed[f]), np.nan_to_num(jump[f])
    X = np.column_stack([
        np.nan_to_num(turn[f]), jp, sp - jp / 2, sp + jp / 2,
        np.minimum(d_take, PROX_CLIP_M), np.minimum(d_other, PROX_CLIP_M),
        sign * nxt1[f], sign * nxt2[f], sign * last1[f], unseen1[f], sign * fv_change[f],
        rule1[f].astype(float)]).astype(np.float32)
    return frames, teams, X


def training_rows(frames, teams, X, gt, dead, fps):
    """Candidates against the team holding the ball (DFL owner just before), in our live play;
    label 1 if the DFL owner changes to the candidate's team within LABEL_S."""
    if not len(frames):
        return X[:0], np.zeros(0)
    before = gt[np.maximum(frames - 3, 0)]
    keep = ~dead[frames] & np.isin(before, (0, 1)) & (before != teams)
    k = int(LABEL_S * fps)
    n = len(gt)
    y = np.zeros(len(frames))
    for i, (f, t) in enumerate(zip(frames, teams)):
        w = gt[max(f - k, 0):min(f + k + 1, n)]
        y[i] = float((w == t).any())
    return X[keep], y[keep]


# Fitted on all seven IDSSE matches (ball hidden like ours, three placements; our dead ball):
# poss_calibrate.changes_shape_run reproduces it. Order: mean, sd, weights (bias, then NAMES).
# Nested leave-one-match-out, with rule 6 on top: spellF1 0.581 against 0.531 for rules 1-4 and
# rule 6, macroF1 0.819 against 0.781, better on all seven; folds chose THRESH 0.25 or 0.3.
MODEL = (
    np.array([41.779, 0.509873, 7.99703, 8.50693, 1.04728, 2.98636, 1.26911, 1.0418, 0.511552, 0.017208, 0.696851, 0.322239]),
    np.array([52.2891, 5.89824, 6.04355, 6.9199, 0.668557, 2.31558, 2.35612, 2.45794, 2.39229, 0.0994551, 2.34394, 0.467341]),
    np.array([-0.807398, 0.216517, -0.0727805, 0.0231983, -0.0417782, -0.446949, 0.174009, -0.438968, 0.700827, 0.0596658, 0.042374, 0.267308, 0.59395]),
)
THRESH = 0.3


def fit(X, y):
    return challenger.fit(X, y)


def prob(model, X):
    return challenger.prob(model, X)


def label(cues, dead, frames, teams, p, thresh, motion=None, p_home=None, q=possession.SHAPE):
    """Codes 0 home / 1 away / 2 dead. Accepted changes replace rule 1's touches; rules 3-6 and
    the control filter as in possession.label."""
    fps, n = cues["fps"], len(dead)
    codes = np.full(n, possession.DEAD, np.int8)
    previous = possession.HOME
    for sg in np.unique(cues["segment"]):
        idx = np.flatnonzero(cues["segment"] == sg)
        a, b = deadball.runs(~dead[idx])
        for s, e in zip(idx[0] + a, idx[0] + b):
            lo, hi = np.searchsorted(frames, s), np.searchsorted(frames, e)
            f, t, pp = frames[lo:hi], teams[lo:hi], p[lo:hi]
            if not len(f):
                previous = 1 - previous
                codes[s:e] = previous
                continue
            owner, at = int(t[0]), s                      # rule 3: first candidate owns the start
            for fr, tm, pr in zip(f[1:], t[1:], pp[1:]):
                if tm != owner and pr >= thresh:
                    codes[at:fr] = owner
                    owner, at = int(tm), int(fr)
            codes[at:e] = owner
            previous = owner
    if p_home is not None:
        codes = possession.shape_override(codes, cues, dead, p_home, q)
    return possession.control_filter(codes, cues["segment"], fps)
