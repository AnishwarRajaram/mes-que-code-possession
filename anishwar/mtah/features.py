"""Per-frame features computed from the tracking. Everything in metres and seconds.

    ball_xy        (n, 2)  ball position, NaN where the tracking lost it
    ball_seen      (n,)    ball position present
    ball_vel       (n, 2)  ball velocity, central difference over +/-1 frame, NaN if a side is missing
    ball_speed     (n,)    |ball_vel|
    turn_deg       (n,)    angle between the ball's path over the TURN_FRAMES before and after
                           this frame -- large at a touch, ~0 in free flight. NaN if unseen.
    speed_jump     (n,)    speed over the next TURN_FRAMES minus speed over the previous ones
    out_of_play    (n,)    ball beyond a touchline or goal line by more than OUT_MARGIN_M
    near_home      (n,)    distance from the ball to the nearest home player (NaN if unseen)
    near_away      (n,)    same for away
    near_home_slot (n,)    slot index of that player, -1 if none
    near_away_slot (n,)
    speed_home     (n,)    mean speed of tracked home players, m/s (from the file's speed column)
    speed_away     (n,)
    n_home, n_away (n,)    tracked players per team

Cached in cache/features.npz (derived data: gitignored, never printed row by row).
"""

import os
import warnings

import numpy as np

from . import config, load

TURN_FRAMES = 3      # 0.12 s each side: long enough to beat position jitter, short enough for a touch
OUT_MARGIN_M = 0.5   # ball centre this far past a line before calling it out (tracking noise)
CACHE_VERSION = 1
NAMES = ("ball_xy", "ball_seen", "ball_vel", "ball_speed", "turn_deg", "speed_jump",
         "out_of_play", "near_home", "near_away", "near_home_slot", "near_away_slot",
         "speed_home", "speed_away", "n_home", "n_away")


def _shift(a, k):
    """a shifted by k frames (positive = look ahead), NaN-padded."""
    out = np.full_like(a, np.nan)
    if k > 0:
        out[:-k] = a[k:]
    elif k < 0:
        out[-k:] = a[:k]
    else:
        out[:] = a
    return out


def _nearest(players_xy, team_mask, ball_xy):
    """Distance to and slot of the nearest tracked player of one team, per frame."""
    diff = players_xy[:, team_mask, :] - ball_xy[:, None, :]
    dist = np.sqrt((diff ** 2).sum(axis=2))
    dist = np.where(np.isnan(dist), np.inf, dist)
    idx = dist.argmin(axis=1)
    best = dist[np.arange(len(dist)), idx]
    slots = np.flatnonzero(team_mask)[idx]
    missing = ~np.isfinite(best)
    return np.where(missing, np.nan, best).astype(np.float32), np.where(missing, -1, slots)


def compute(atd):
    length, width = config.PITCH
    ball_xy = load.to_metres(atd["ball"][:, :2])
    players_xy = load.to_metres(atd["players"][:, :, :2])
    team = atd["team"]
    home, away = team == load.HOME, team == load.AWAY

    ball_seen = ~np.isnan(ball_xy[:, 0])
    ball_vel = (_shift(ball_xy, 1) - _shift(ball_xy, -1)) * (config.FPS / 2.0)
    ball_speed = np.linalg.norm(ball_vel, axis=1)

    before = ball_xy - _shift(ball_xy, -TURN_FRAMES)
    after = _shift(ball_xy, TURN_FRAMES) - ball_xy
    norm_b, norm_a = np.linalg.norm(before, axis=1), np.linalg.norm(after, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        cos = (before * after).sum(axis=1) / (norm_b * norm_a)
        turn_deg = np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))
    # A ball standing still has no direction; call it no turn rather than NaN.
    still = ball_seen & ((norm_b < 0.05) | (norm_a < 0.05))
    turn_deg[still] = 0.0
    speed_jump = (norm_a - norm_b) * (config.FPS / float(TURN_FRAMES))

    x, y = ball_xy[:, 0], ball_xy[:, 1]
    with np.errstate(invalid="ignore"):
        out_of_play = ball_seen & ((x < -OUT_MARGIN_M) | (x > length + OUT_MARGIN_M)
                                   | (y < -OUT_MARGIN_M) | (y > width + OUT_MARGIN_M))

    near_home, near_home_slot = _nearest(players_xy, home, ball_xy)
    near_away, near_away_slot = _nearest(players_xy, away, ball_xy)

    speeds = atd["players"][:, :, 2]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN frames -> NaN, as wanted
        speed_home = np.nanmean(speeds[:, home], axis=1)
        speed_away = np.nanmean(speeds[:, away], axis=1)
    present = ~np.isnan(players_xy[:, :, 0])

    return {
        "ball_xy": ball_xy.astype(np.float32), "ball_seen": ball_seen,
        "ball_vel": ball_vel.astype(np.float32), "ball_speed": ball_speed.astype(np.float32),
        "turn_deg": turn_deg.astype(np.float32), "speed_jump": speed_jump.astype(np.float32),
        "out_of_play": out_of_play, "near_home": near_home, "near_away": near_away,
        "near_home_slot": near_home_slot.astype(np.int16),
        "near_away_slot": near_away_slot.astype(np.int16),
        "speed_home": speed_home.astype(np.float32), "speed_away": speed_away.astype(np.float32),
        "n_home": present[:, home].sum(axis=1).astype(np.int8),
        "n_away": present[:, away].sum(axis=1).astype(np.int8),
    }


def load_features(rebuild=False):
    """Features for the tactical source, from the cache, building them first if needed."""
    path = os.path.join(config.CACHE_DIR, "features.npz")
    atd_path = os.path.join(config.CACHE_DIR, "tactical_atd.npz")
    if not rebuild and os.path.exists(path) and os.path.exists(atd_path) \
            and os.path.getmtime(path) >= os.path.getmtime(atd_path):
        with np.load(path) as cached:
            if int(cached["version"]) == CACHE_VERSION:
                return {name: cached[name] for name in NAMES}

    feats = compute(load.load_atd("tactical"))
    os.makedirs(config.CACHE_DIR, exist_ok=True)
    np.savez(path, version=CACHE_VERSION, **feats)
    return feats
