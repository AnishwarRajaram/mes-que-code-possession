"""Dead-ball detection from tracking alone, built the same way on our match and on other matches.

Three parts:

  build_cues   per-frame cues from positions only (players' mean speed, its ratio to the local
               5-minute norm, metres the ball is past a line), identical for both data sources
  dropout      hides the ball on another match the way our tactical tracking loses it: in long
               blocks, at our rates separately in dead and in live play (as our own rules call
               them on both matches), more often while players are slow. The profile is measured
               on our match's tracking (no labels), see ball_gap_profile()
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
  4. Inferred ball-out: tracking often loses the ball right at the line. If the ball was last seen
     within OUT_NEAR_M of a line, moving towards it, and then stays unseen for OUT_GAP_S, it is
     treated as out from the moment it vanished (then rule 3 applies).
  5. Kick-off shape (Law 8): every player in their own half (bar the kicker, who straddles the line) and one player at the centre spot. Held
     for KO_MIN_S, play is dead until the shape breaks.
"""

import numpy as np
import pandas as pd

NORM_WINDOW_S = 300     # local norm of the match tempo: 5 minutes, long against any stoppage
SMOOTH_S = 1.0          # players' speed smoothed over 1 s, the length of a stride cycle
MAX_PLAYER_SPEED = 12.0  # faster than any sprinter: a tracking jump, not a player

# Calibrated on the seven IDSSE matches with the ball hidden like ours (calibrate.py, which gives
# back exactly these values, also after re-measuring the hiding profile with them): best on all
# seven, 17 of 18 one-step neighbours within 0.01. Nested leave-one-match-out: stoppage F1
# 0.756 +- 0.034, frame F1 0.865 +- 0.017. Every fold turned rule 4 (inferred ball-out) off, so
# OUT_GAP_S has no effect; KO_MIN_S made no difference on IDSSE. Both sit at the bottom of their
# ranges by the search's tie rule.
PARAMS = {
    "SLOW": 1.0,           # players at or below their usual pace ...
    "RESUME": 1.15,        # ... and staying under 1.15x it ...
    "MIN_STOP_S": 9.0,     # ... for 9 s is a stoppage
    "START_BACK_S": -0.5,  # the stoppage starts 0.5 s into the stretch below RESUME
    "END_LEAD_S": 1.0,     # players speed up about 1 s before the restart is taken
    "OUT_LINK_S": 12.0,    # a ball-out belongs to the stoppage that starts within 12 s
    "OUT_MIN_S": 8.0,      # a ball-out with no stoppage after it is dead for 8 s
    "OUT_NEAR_M": 0.0,     # inferred ball-out: last seen this close to a line (0 = rule off) ...
    "OUT_GAP_S": 0.5,      # ... then unseen for this long
    "KO_SHARE": 0.85,      # kick-off shape: share of each team in its own half (>1 = rule off)
    "KO_MIN_S": 0.5,       # ... held this long
}
KO_CENTRE_M = 3.0          # a player this close to the centre spot (Law 8: the kicker is on it)
LAST_SEEN_FRAMES = 3       # ball heading measured over the last 3 frames before it vanished


def runs(mask):
    """[(start, stop)] of True runs, stop exclusive."""
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)


def _speed(xy, fps):
    v = np.full(xy.shape[:-1], np.nan, np.float32)
    d = (xy[2:] - xy[:-2]) * (fps / 2.0)
    v[1:-1] = np.sqrt((d ** 2).sum(-1))
    return v


def kickoff_shape(players_xy, team, pitch):
    """Per frame: how completely the two teams sit in opposite halves (the smaller of the two
    teams' own-half shares, 0 if they are not on opposite sides), and whether a player stands
    within KO_CENTRE_M of the centre spot."""
    length, width = pitch
    present = ~np.isnan(players_xy[:, :, 0])
    left = players_xy[:, :, 0] < length / 2
    shares = []
    for t in (0, 1):
        on = present & (team == t)[None, :]
        shares.append((left & on).sum(1) / np.maximum(on.sum(1), 1))
    h, a = shares
    side = np.where(h >= a, np.minimum(h, 1 - a), np.minimum(1 - h, a))
    centre = np.sqrt((players_xy[:, :, 0] - length / 2) ** 2 + (players_xy[:, :, 1] - width / 2) ** 2)
    with np.errstate(invalid="ignore"):
        at_spot = np.nanmin(np.where(present, centre, np.inf), axis=1) <= KO_CENTRE_M
    return side.astype(np.float32), at_spot


def build_cues(ball_xy, players_xy, pitch, fps, segment=None, team=None):
    """Per-frame cues. ball_xy (n,2) and players_xy (n,p,2) in metres, NaN where untracked;
    team (p,) 0 home / 1 away; segment (n,) breaks the smoothing and norm at half-time."""
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
    side, at_spot = kickoff_shape(players_xy, team, pitch)
    near = []
    for t in (0, 1):                     # each team's nearest player to the ball, metres
        cols = np.flatnonzero(team == t)
        d = np.sqrt(((players_xy[:, cols, :] - ball_xy[:, None, :]) ** 2).sum(-1))
        with np.errstate(invalid="ignore"):
            near.append(np.nanmin(np.where(np.isnan(d), np.inf, d), axis=1).astype(np.float32))
    near = [np.where(np.isfinite(v), v, np.nan) for v in near]
    return {"speed": speed, "ratio": ratio, "beyond": beyond, "ball_seen": ~np.isnan(x),
            "ball_xy": ball_xy.astype(np.float32), "ko_side": side, "ko_spot": at_spot,
            "near_home": near[0], "near_away": near[1],
            "pitch": np.asarray(pitch, np.float32), "segment": segment, "fps": fps}


BANDS = np.array([0, 0.5, 0.65, 0.8, 1.0, 1.2, 1.5, np.inf])


def _band(ratio):
    return np.clip(np.searchsorted(BANDS, np.nan_to_num(ratio, nan=1.0), side="right") - 1,
                   0, len(BANDS) - 2)


def ball_gap_profile(cues, graded):
    """How the ball goes missing on a match, as aggregates, separately where our own rules call
    the ball dead and where they call it live: the missing share, the gap lengths (s, each gap cut
    at the edge of that state) and the missing rate in each band of the speed ratio. Measured on
    tracking only, no labels; the same rules run on the other match before hiding its ball."""
    fps = cues["fps"]
    dead = detect(cues)
    miss = ~cues["ball_seen"]
    band = _band(cues["ratio"])
    out = {"bands": BANDS}
    for name, state in (("dead", dead), ("live", ~dead)):
        here = graded & state
        a, b = runs(miss & here)
        lengths = (b - a) / fps
        rate = np.array([miss[here & (band == k)].mean() if (here & (band == k)).any() else 0.0
                         for k in range(len(BANDS) - 1)])
        out[name] = {"share": float(miss[here].mean()), "gaps_s": lengths[lengths >= 1.0],
                     "rate": rate}
    return out


def dropout(cues, profile, seed=0):
    """A copy of cues with the ball hidden the way ours goes missing: our dead-ball rules run on
    this match first, then, separately in its dead and its live stretches, gaps of our lengths are
    placed (more often where players are slow, as on ours) until each state's missing share
    matches ours. A gap never runs past the edge of its state."""
    fps = cues["fps"]
    n = len(cues["ratio"])
    rng = np.random.default_rng(seed)
    dead = detect(cues)
    band = _band(cues["ratio"])
    hidden = ~cues["ball_seen"].copy()
    for name, state in (("dead", dead), ("live", ~dead)):
        prof = profile[name]
        frames = np.flatnonzero(state)
        if not len(frames) or not len(prof["gaps_s"]):
            continue
        a, b = runs(state)
        run_end = np.zeros(n, np.int64)
        for x, y in zip(a, b):
            run_end[x:y] = y
        accept = prof["rate"][band] / max(prof["rate"].max(), 1e-9)
        while hidden[frames].mean() < prof["share"]:
            start = int(frames[rng.integers(0, len(frames))])
            if rng.random() > accept[start]:
                continue
            length = int(rng.choice(prof["gaps_s"]) * fps)
            hidden[start:min(start + length, run_end[start])] = True
    out = dict(cues)
    out["ball_seen"] = ~hidden
    out["beyond"] = np.where(hidden, np.nan, cues["beyond"])
    out["ball_xy"] = np.where(hidden[:, None], np.nan, cues["ball_xy"])
    for k in ("near_home", "near_away"):
        if k in cues:
            out[k] = np.where(hidden, np.nan, cues[k])
    return out


def inferred_outs(cues, p=PARAMS):
    """Frames where the ball vanished near a line, heading for it, and stayed unseen OUT_GAP_S."""
    if p["OUT_NEAR_M"] <= 0:
        return np.zeros(0, int)
    fps = cues["fps"]
    length, width = cues["pitch"]
    xy = cues["ball_xy"]
    a, b = runs(~cues["ball_seen"])
    keep = (a > LAST_SEEN_FRAMES) & ((b - a) >= p["OUT_GAP_S"] * fps)
    a = a[keep]
    last, before = xy[a - 1], xy[a - 1 - LAST_SEEN_FRAMES]
    v = last - before
    # distance to each line, and whether the ball was moving towards it
    dist = np.stack([last[:, 0], length - last[:, 0], last[:, 1], width - last[:, 1]], 1)
    towards = np.stack([v[:, 0] < 0, v[:, 0] > 0, v[:, 1] < 0, v[:, 1] > 0], 1)
    with np.errstate(invalid="ignore"):
        hit = ((dist <= p["OUT_NEAR_M"]) & towards).any(1)
    return a[np.nan_to_num(hit, nan=0).astype(bool)]


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
    oa = np.union1d(oa, inferred_outs(cues, p))                 # rule 4
    stop_starts = np.array([s for s, _ in stops]) if stops else np.zeros(0, int)
    for o in oa:
        later = stop_starts[(stop_starts >= o - int(p["START_BACK_S"] * fps))
                            & (stop_starts <= o + int(p["OUT_LINK_S"] * fps))]
        if len(later):
            k = int(np.flatnonzero(stop_starts == later[0])[0])
            stops[k] = (min(stops[k][0], o), stops[k][1])
        else:
            stops.append((o, o + int(p["OUT_MIN_S"] * fps)))
    # Rule 5: kick-off shape.
    if p["KO_SHARE"] <= 1:
        ka, kb = runs((cues["ko_side"] >= p["KO_SHARE"]) & cues["ko_spot"])
        stops += [(s, e) for s, e in zip(ka, kb) if e - s >= p["KO_MIN_S"] * fps]
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
