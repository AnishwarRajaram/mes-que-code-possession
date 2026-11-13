"""Dead-ball detection from tracking alone, built the same way on our match and on other matches.

Three parts:

  build_cues   per-frame cues from positions only (players' mean speed, its ratio to the local
               5-minute norm, metres the ball is past a line), identical for both data sources
  dropout      hides the ball on another match the way our tactical tracking loses it: in long
               blocks, more often while players are slow. The profile is measured on our match's
               tracking (no labels), see ball_gap_profile()
  detect       the rules. Every constant is in PARAMS and is calibrated on other matches with
               labels (IDSSE, DFL ball status), leave-one-match-out, never on the provided game

Rules, in words (Laws of the Game, Law 9: the ball is out when it wholly crosses a line, or play
is stopped by the referee):
  1. Ball past a line: dead from the moment it crosses.
  2. Players stop: when the players' mean speed falls well below the local norm (SLOW) and stays
     there for MIN_STOP_S, play is dead. The stoppage spans the whole stretch below RESUME, the
     start pulled back by START_BACK_S (players slow down a few seconds after the whistle) and the
     end pulled back by END_LEAD_S (players speed up just before the restart is taken).
  3. A ball-out joins the players' stop that follows within OUT_LINK_S; a ball-out with no stop
     after it (a quick throw-in) is dead for OUT_MIN_S.
"""

import numpy as np
import pandas as pd

NORM_WINDOW_S = 300     # local norm of the match tempo: 5 minutes, long against any stoppage
SMOOTH_S = 1.0          # players' speed smoothed over 1 s, the length of a stride cycle
MAX_PLAYER_SPEED = 12.0  # faster than any sprinter: a tracking jump, not a player

# Calibrated on the seven IDSSE matches with the ball hidden like ours (calibrate.py): best on all
# seven, every one-step neighbour within 0.01, none at the edge of its range. Nested
# leave-one-match-out: stoppage F1 0.740 +- 0.018, frame F1 0.855 +- 0.020.
PARAMS = {
    "SLOW": 0.95,          # players at or below their usual pace ...
    "RESUME": 1.15,        # ... and staying under 1.15x it ...
    "MIN_STOP_S": 8.0,     # ... for 8 s is a stoppage
    "START_BACK_S": -0.5,  # the stretch below RESUME already starts at the whistle, or just before
    "END_LEAD_S": 1.0,     # players speed up about 1 s before the restart is taken
    "OUT_LINK_S": 12.0,    # a ball-out belongs to the stoppage that starts within 12 s
    "OUT_MIN_S": 8.0,      # a ball-out with no stoppage after it is dead for 8 s
}


def runs(mask):
    """[(start, stop)] of True runs, stop exclusive."""
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)


def _speed(xy, fps):
    v = np.full(xy.shape[:-1], np.nan, np.float32)
    d = (xy[2:] - xy[:-2]) * (fps / 2.0)
    v[1:-1] = np.sqrt((d ** 2).sum(-1))
    return v


def build_cues(ball_xy, players_xy, pitch, fps, segment=None):
    """Per-frame cues. ball_xy (n,2) and players_xy (n,p,2) in metres, NaN where untracked;
    segment (n,) breaks the smoothing and norm at half-time."""
    length, width = pitch
    n = len(ball_xy)
    segment = np.zeros(n, np.int8) if segment is None else segment
    with np.errstate(invalid="ignore"):
        ps = _speed(players_xy, fps)
        mean_speed = np.nanmean(np.where(ps < MAX_PLAYER_SPEED, ps, np.nan), axis=1)
        x, y = ball_xy[:, 0], ball_xy[:, 1]
        beyond = np.maximum.reduce([-x, x - length, -y, y - width])
    speed = np.full(n, np.nan)
    ratio = np.full(n, np.nan)
    for s in np.unique(segment):
        idx = np.flatnonzero(segment == s)
        sm = pd.Series(mean_speed[idx]).rolling(int(SMOOTH_S * fps), center=True,
                                                min_periods=1).mean()
        per_s = sm.groupby(np.arange(len(sm)) // fps).mean()
        norm = per_s.rolling(NORM_WINDOW_S, center=True, min_periods=30).median().bfill().ffill()
        norm = np.repeat(norm.values, fps)[:len(sm)]
        speed[idx] = sm.values
        ratio[idx] = sm.values / norm
    return {"speed": speed, "ratio": ratio, "beyond": beyond, "ball_seen": ~np.isnan(x),
            "segment": segment, "fps": fps}


def ball_gap_profile(cues, graded):
    """How the ball goes missing on a match, as aggregates: missing share, gap lengths (s) and
    the missing rate in each band of the speed ratio. Measured on tracking only, no labels."""
    fps = cues["fps"]
    miss = ~cues["ball_seen"] & graded
    a, b = runs(miss)
    lengths = (b - a) / fps
    bands = np.array([0, 0.5, 0.65, 0.8, 1.0, 1.2, 1.5, np.inf])
    r = cues["ratio"][graded]
    m = miss[graded]
    band = np.clip(np.searchsorted(bands, r, side="right") - 1, 0, len(bands) - 2)
    rate = np.array([m[band == k].mean() if (band == k).any() else 0.0
                     for k in range(len(bands) - 1)])
    return {"share": float(m.mean()), "gaps_s": lengths[lengths >= 1.0], "bands": bands,
            "rate": rate}


def dropout(cues, profile, seed=0):
    """A copy of cues with the ball hidden in blocks drawn from profile: gap lengths resampled
    from ours, placed more often where players are slow, until the missing share matches."""
    fps = cues["fps"]
    n = len(cues["ratio"])
    rng = np.random.default_rng(seed)
    band = np.clip(np.searchsorted(profile["bands"], np.nan_to_num(cues["ratio"], nan=1.0),
                                   side="right") - 1, 0, len(profile["rate"]) - 1)
    accept = profile["rate"][band] / profile["rate"].max()
    hidden = ~cues["ball_seen"].copy()
    target = profile["share"]
    while hidden.mean() < target:
        start = int(rng.integers(0, n))
        if rng.random() > accept[start]:
            continue
        length = int(rng.choice(profile["gaps_s"]) * fps)
        hidden[start:start + length] = True
    out = dict(cues)
    out["ball_seen"] = ~hidden
    out["beyond"] = np.where(hidden, np.nan, cues["beyond"])
    return out


def detect(cues, p=PARAMS):
    """Per-frame dead mask from the rules above."""
    fps = cues["fps"]
    n = len(cues["ratio"])
    ratio = np.nan_to_num(cues["ratio"], nan=1.0)
    dead = np.zeros(n, bool)

    # Rule 2: players stop.
    below = ratio < p["RESUME"]
    core = ratio < p["SLOW"]
    a, b = runs(below)
    stops = []
    for s, e in zip(a, b):
        # needs MIN_STOP_S of continuous SLOW inside the stretch
        ca, cb = runs(core[s:e])
        if not len(ca) or (cb - ca).max() < p["MIN_STOP_S"] * fps:
            continue
        s2 = max(s - int(p["START_BACK_S"] * fps), 0)
        e2 = max(e - int(p["END_LEAD_S"] * fps), s2 + 1)
        stops.append((s2, e2))

    # Rule 1 and 3: ball past a line.
    out = np.nan_to_num(cues["beyond"], nan=-1.0) > 0
    oa, _ = runs(out)
    stop_starts = np.array([s for s, _ in stops]) if stops else np.zeros(0, int)
    for o in oa:
        later = stop_starts[(stop_starts >= o - int(p["START_BACK_S"] * fps))
                            & (stop_starts <= o + int(p["OUT_LINK_S"] * fps))]
        if len(later):
            k = int(np.flatnonzero(stop_starts == later[0])[0])
            stops[k] = (min(stops[k][0], o), stops[k][1])
        else:
            stops.append((o, o + int(p["OUT_MIN_S"] * fps)))
    for s, e in stops:
        dead[s:min(e, n)] = True
    return dead


def score(dead_pred, dead_true, segment, fps, iou=0.5):
    """Frame F1 of dead, stoppage F1 (stoppages of at least 1 s matched one-to-one at
    temporal IoU >= iou, the spellF1 idea applied to stoppages), dead share, stoppages per 90."""
    tp = (dead_pred & dead_true).sum()
    f1_frame = 2 * tp / max(dead_pred.sum() + dead_true.sum(), 1)

    def stoppages(mask):
        out = []
        for sg in np.unique(segment):
            idx = np.flatnonzero(segment == sg)
            a, b = runs(mask[idx])
            out += [(idx[0] + s, idx[0] + e) for s, e in zip(a, b) if e - s >= fps]
        return out

    P, T = stoppages(dead_pred), stoppages(dead_true)
    matched, j = 0, 0
    used = set()
    for s, e in P:
        while j < len(T) and T[j][1] <= s:
            j += 1
        k = j
        while k < len(T) and T[k][0] < e:
            ts, te = T[k]
            inter = min(e, te) - max(s, ts)
            if k not in used and inter / (max(e, te) - min(s, ts)) >= iou:
                used.add(k)
                matched += 1
                break
            k += 1
    f1_stop = 2 * matched / max(len(P) + len(T), 1)
    minutes = len(dead_true) / fps / 60
    return {"f1_frame": float(f1_frame), "f1_stop": float(f1_stop),
            "dead_share": float(dead_pred.mean()), "stops_per90": len(P) * 90 / minutes}
