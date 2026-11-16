"""Who has the ball when the ball is not tracked: read it from the players.

A team in possession moves forward and spreads out to make the pitch big; the team without it
drops back and squeezes. Those shapes are visible in the players alone. Features are computed per
frame, the same way on our match and on other matches, always as home minus away so that the
model has one symmetric job (probability that home has the ball):

  forward velocity   median velocity of the team's players towards the goal it attacks (m/s)
  forward position   team's median distance past halfway, towards the goal it attacks (m)
  depth, width       interquartile spread of the team along and across the pitch (m)

Medians and interquartile ranges, not means and standard deviations: our automatic tracking has
stray tracks that inflated a standard deviation 2.5x against IDSSE's.
  in the other half  share of the team's players past halfway
  last owner         who had the ball when it was last seen, faded with time (prior)

Attack direction per half: the team whose players sit, on median, nearer the left goal attacks
right. The model is a logistic regression (challenger.fit), trained on IDSSE with the DFL owner
as the label, only on live frames where the ball is hidden; it never sees the provided game's
labels.
"""

import numpy as np
import pandas as pd

from . import challenger

SMOOTH_S = 1.0
PRIOR_FADE_S = 6.0
MAX_SPEED = 12.0     # m/s, as in deadball.MAX_PLAYER_SPEED: faster is a tracking jump
PRESS_R = 5.0        # opponents this close count as pressing (fixed in advance, not tuned)
PRESS_CLIP_M = 20.0  # nearest-opponent distances beyond this carry no pressure information
SHAPE_NAMES = ("forward velocity", "forward position", "depth", "width", "in the other half")
PRESS_NAMES = ("pressers on most-pressed player", "nearest opponent to them",
               "opponents closing on them")
PRIOR_NAMES = ("last owner (faded)", "last owner x forward velocity")
NAMES = SHAPE_NAMES + PRIOR_NAMES


def names(n_inputs):
    """Input names for a model with n_inputs columns (with or without the pressure features)."""
    return SHAPE_NAMES + (PRESS_NAMES if n_inputs > len(NAMES) else ()) + PRIOR_NAMES


def _smooth(a, fps):
    return pd.Series(a).rolling(int(SMOOTH_S * fps), center=True, min_periods=1).mean().values


def team_features(players_xy, team, pitch, fps, segment):
    """(n, 5) home-minus-away shape features, from positions only."""
    length, _ = pitch
    n = len(players_xy)
    v = np.full_like(players_xy, np.nan)
    v[1:-1] = (players_xy[2:] - players_xy[:-2]) * (fps / 2.0)
    with np.errstate(invalid="ignore"):
        v = np.where(np.linalg.norm(v, axis=-1, keepdims=True) < 12, v, np.nan)
    out = np.zeros((n, 5), np.float32)
    direction = np.ones((n, 2))
    for sg in np.unique(segment):
        idx = np.flatnonzero(segment == sg)
        med = [np.nanmedian(players_xy[idx][:, team == t, 0]) for t in (0, 1)]
        d = 1.0 if med[0] < med[1] else -1.0
        direction[idx, 0], direction[idx, 1] = d, -d
    per_team = []
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for t in (0, 1):
            cols = team == t
            x, y = players_xy[:, cols, 0], players_xy[:, cols, 1]
            dt = direction[:, t]
            # medians and interquartile ranges: one stray track must not move a team's shape
            fwd_v = np.nanmedian(v[:, cols, 0], axis=1) * dt
            fwd_x = (np.nanmedian(x, axis=1) - length / 2) * dt
            qx, qy = np.nanpercentile(x, (25, 75), axis=1), np.nanpercentile(y, (25, 75), axis=1)
            depth = qx[1] - qx[0]
            width = qy[1] - qy[0]
            other = np.nanmean(((x - length / 2) * dt[:, None]) > 0, axis=1)
            per_team.append(np.column_stack([fwd_v, fwd_x, depth, width, other]))
    diff = np.nan_to_num(per_team[0] - per_team[1])
    for j in range(diff.shape[1]):
        out[:, j] = _smooth(diff[:, j], fps)
    return out


def pressure_features(players_xy, team, fps, chunk=20000):
    """(n, 3) home-minus-away pressure on each team's most-pressed player, from positions only.

    Defenders close in on whoever has the ball, so the player under most pressure is a strong
    hint of the owner even when the ball is not tracked. Per team and frame, the most-pressed
    player is the one with most opponents within PRESS_R, ties going to the nearer opponent.
    Features: that count, the distance to that player's nearest opponent (clipped at
    PRESS_CLIP_M), and how fast that distance is shrinking (m/s)."""
    n = len(players_xy)
    xy = players_xy.astype(np.float64)
    h, a = xy[:, team == 0], xy[:, team == 1]
    near_h = np.full((n, h.shape[1]), np.inf)       # each home player's nearest away player
    near_a = np.full((n, a.shape[1]), np.inf)
    cnt_h = np.zeros((n, h.shape[1]))
    cnt_a = np.zeros((n, a.shape[1]))
    for s in range(0, n, chunk):
        d = np.linalg.norm(h[s:s + chunk, :, None, :] - a[s:s + chunk, None, :, :], axis=-1)
        d = np.where(np.isnan(d), np.inf, d)
        near_h[s:s + chunk], near_a[s:s + chunk] = d.min(axis=2), d.min(axis=1)
        cnt_h[s:s + chunk], cnt_a[s:s + chunk] = (d <= PRESS_R).sum(axis=2), (d <= PRESS_R).sum(axis=1)
    per_team = []
    rows = np.arange(n)
    for near, cnt in ((near_h, cnt_h), (near_a, cnt_a)):
        near = np.minimum(near, PRESS_CLIP_M)
        closing = np.zeros_like(near)
        closing[1:-1] = -(near[2:] - near[:-2]) * (fps / 2.0)
        closing = np.clip(closing, -MAX_SPEED, MAX_SPEED)
        pick = np.argmax(cnt - near / (PRESS_CLIP_M + 1.0), axis=1)   # most pressers, then nearest
        per_team.append(np.column_stack([cnt[rows, pick], near[rows, pick], closing[rows, pick]]))
    diff = per_team[0] - per_team[1]
    out = np.zeros((n, 3), np.float32)
    for j in range(3):
        out[:, j] = _smooth(diff[:, j], fps)
    return out


def with_prior(feat, cues, fps):
    """Adds the last owner seen (+1 home, -1 away, 0 unknown), faded since the ball vanished, and
    its product with forward velocity. Uses the touches the possession rules already found."""
    seen = cues["ball_seen"]
    n = len(seen)
    last_seen = np.maximum.accumulate(np.where(seen, np.arange(n), -1))
    dt = np.where(last_seen >= 0, (np.arange(n) - last_seen) / fps, 1e9)
    owner = cues.get("_last_owner", np.zeros(n))
    prior = owner * np.exp(-dt / PRIOR_FADE_S)
    return np.column_stack([feat, prior, prior * feat[:, 0]]).astype(np.float32)


def inputs(cues, codes0, press=False):
    """Model inputs: shape features (and, with press, the pressure features) plus the faded last
    owner from a first pass of the rules."""
    seen = cues["ball_seen"]
    n = len(seen)
    val = np.where(codes0 == 0, 1.0, np.where(codes0 == 1, -1.0, 0.0))
    last = np.maximum.accumulate(np.where(seen, np.arange(n), -1))
    w = dict(cues)
    w["_last_owner"] = np.where(last >= 0, val[np.maximum(last, 0)], 0.0)
    feat = np.column_stack([cues["shape"], cues["press"]]) if press else cues["shape"]
    return with_prior(feat, w, cues["fps"])


# Fitted on all seven IDSSE matches (smoothed like our tracking, ball hidden like ours, three
# placements, live frames without the ball, DFL owner as label, 5 frames a second), on the change
# detector's first pass (changes.py): poss_calibrate.changes_shape_run reproduces it. Order: mean,
# sd, then weights (bias first, then NAMES).
MODEL = (
    np.array([0.10349, 6.28007, 1.31246, 0.553315, 0.0531001, 0.0472813, 0.574993]),
    np.array([3.52672, 34.2707, 8.12996, 5.49811, 0.309649, 0.545632, 1.80609]),
    np.array([0.278661, 0.873143, -1.32478, 0.749537, 0.957446, 0.885292, 0.729753, 0.0268243]),
)


def fit(X, y):
    return challenger.fit(X, y)


def prob(model, X):
    return challenger.prob(model, X)
