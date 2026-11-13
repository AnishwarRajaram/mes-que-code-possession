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
NAMES = ("forward velocity", "forward position", "depth", "width", "in the other half",
         "last owner (faded)", "last owner x forward velocity")


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


def inputs(cues, codes0):
    """Model inputs: shape features plus the faded last owner from a first pass of the rules."""
    seen = cues["ball_seen"]
    n = len(seen)
    val = np.where(codes0 == 0, 1.0, np.where(codes0 == 1, -1.0, 0.0))
    last = np.maximum.accumulate(np.where(seen, np.arange(n), -1))
    w = dict(cues)
    w["_last_owner"] = np.where(last >= 0, val[np.maximum(last, 0)], 0.0)
    return with_prior(cues["shape"], w, cues["fps"])


# Fitted on all seven IDSSE matches (ball hidden like ours, three placements, live frames without
# the ball, DFL owner as label, 5 frames a second): poss_calibrate.shape_fit reproduces it.
# Order: mean, sd, then weights (bias first, then NAMES). Nested leave-one-match-out owner
# accuracy 83.2% on live frames without the ball (holding the last owner: 75.6%).
MODEL = (
    np.array([-0.0893369, 5.76451, 0.966337, 0.455832, 0.0484894, 0.0322447, 0.534331]),
    np.array([3.51897, 35.0312, 8.16746, 5.50758, 0.315507, 0.558993, 1.80834]),
    np.array([0.0915354, 0.904832, -1.44306, 0.786246, 0.983624, 1.05135, 0.842327, 0.149746]),
)


def fit(X, y):
    return challenger.fit(X, y)


def prob(model, X):
    return challenger.prob(model, X)
