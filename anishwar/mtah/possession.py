"""Who has the ball while it is alive: touches, held possession, a control filter, restarts.

CHALLENGE.md: possession changes when the other team gains *controlled* possession; while the ball
is loose it stays with the last team; a touch, a deflection or a won header without control does
not count. On other matches (IDSSE) the DFL's BallPossession changes exactly at the new team's
first touch (nearest player a median 0.6 m from the ball, at the moment of the change), so it is a
last-touch record. Both the target and our prediction go through the same CONTROL filter to turn
last touch into controlled possession.

Rules:
  1. Touch: a team's nearest player within TOUCH_R of the ball, the other team's nearest at least
     MARGIN_M further away, and the ball doing something a touch does -- turning by TURN_DEG,
     changing speed by JUMP_MS, or moving slower than CARRY_MS (at the feet).
  2. Hold: possession is the team of the last touch, through loose play and through frames where
     the ball is not tracked (writing the last team costs exactly what X costs on spellF1).
  3. Restart: each live stretch starts with the team of its first touch. A live stretch with no
     touch at all goes to the team that did not have the ball before the stoppage.
  4. Hidden change: when the ball was out of sight between two touches of different teams, the
     change is placed GAP_FRAC of the way through the gap instead of at the second touch.
  6. Shape (shape.py): inside a ball gap, the players show who has it; see shape_override.
  5. Control: a spell shorter than CONTROL_S between two spells of the other team is a touch
     without control and goes back to that team. CONTROL_S is fixed from the challenge's own
     words ("a player keeps the ball for a few seconds") -- it is never tuned, and never chosen
     on the provided game.
"""

import numpy as np

from . import deadball

HOME, AWAY, DEAD = 0, 1, 2
CONTROL_S = 2.0
TURN_FRAMES = 3      # ball direction compared over 0.12 s either side

# Calibrated end to end on the seven IDSSE matches (poss_calibrate.py): our dead-ball rules, the
# ball hidden like ours, official spellF1 (poss_calibrate.py gives back exactly these values). Best
# on all seven, all 10 one-step neighbours within 0.01. Nested leave-one-match-out: spellF1
# 0.513 +- 0.024, macroF1 0.770 +- 0.006.
PARAMS = {
    "TOUCH_R": 1.5,     # a player within 1.5 m of the ball ...
    "MARGIN_M": 0.25,   # ... and the nearest opponent at least 0.25 m further away ...
    "TURN_DEG": 20.0,   # ... while the ball turns by 20 degrees or more ...
    "JUMP_MS": 13.0,    # ... or changes speed by 13 m/s (a kick or a stop)
    "CARRY_MS": 0.0,    # slow ball at the feet: rejected (0 = off), it adds false touches in duels
    "GAP_FRAC": 1.0,    # a change hidden by a ball gap is placed at its end
}


def _shift(a, k):
    out = np.full_like(a, np.nan)
    if k > 0:
        out[:-k] = a[k:]
    elif k < 0:
        out[-k:] = a[:k]
    else:
        out[:] = a
    return out


def ball_motion(cues):
    """Ball speed (m/s), turn (degrees) and speed change (m/s) around each frame; NaN if unseen."""
    fps, xy = cues["fps"], cues["ball_xy"].astype(np.float64)
    k = TURN_FRAMES
    before, after = xy - _shift(xy, -k), _shift(xy, k) - xy
    nb, na = np.linalg.norm(before, axis=1), np.linalg.norm(after, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        cos = (before * after).sum(1) / (nb * na)
        turn = np.degrees(np.arccos(np.clip(cos, -1, 1)))
    turn[(nb < 0.05) | (na < 0.05)] = 0.0
    speed = (nb + na) * fps / (2 * k)
    jump = (na - nb) * fps / k
    return speed, turn, jump


def touches(cues, p=PARAMS, motion=None):
    """(frames, teams) of touches."""
    speed, turn, jump = ball_motion(cues) if motion is None else motion
    dh = np.nan_to_num(cues["near_home"], nan=np.inf)
    da = np.nan_to_num(cues["near_away"], nan=np.inf)
    with np.errstate(invalid="ignore"):
        acts = (turn >= p["TURN_DEG"]) | (np.abs(jump) >= p["JUMP_MS"]) | (speed < p["CARRY_MS"])
    acts &= cues["ball_seen"]
    home = (dh <= p["TOUCH_R"]) & (da >= dh + p["MARGIN_M"]) & acts
    away = (da <= p["TOUCH_R"]) & (dh >= da + p["MARGIN_M"]) & acts
    frames = np.flatnonzero(home | away)
    return frames, np.where(home[frames], HOME, AWAY)


def control_filter(codes, segment, fps, control_s=CONTROL_S):
    """Spells shorter than control_s sandwiched between two spells of the other team go back to
    it. Applied the same way to the target and to the prediction."""
    out = codes.copy()
    for sg in np.unique(segment):
        idx = np.flatnonzero(segment == sg)
        c = out[idx]
        b = np.flatnonzero(np.diff(c) != 0) + 1
        starts, stops = np.r_[0, b], np.r_[b, len(c)]
        kinds = c[starts].copy()
        for i in range(1, len(starts) - 1):
            if kinds[i] in (HOME, AWAY) and kinds[i - 1] == kinds[i + 1] \
                    and kinds[i - 1] in (HOME, AWAY) and kinds[i - 1] != kinds[i] \
                    and (stops[i] - starts[i]) < control_s * fps:
                kinds[i] = kinds[i - 1]
                c[starts[i]:stops[i]] = kinds[i]
        out[idx] = c
    return out


# Chosen with the change detector (changes.py) inside each fold of poss_calibrate.changes_shape_run:
# every fold 0.45, held 4 s (five folds) or 2 s (two); 2 s on all seven. Averaging first (SMOOTH_S) added nothing.
SHAPE = {
    "SWITCH_P": 0.45,   # inside a ball gap, the other team takes over when the players' shape says
    "SWITCH_S": 2.0,   # so with this probability, held for this long (shape.py) ...
    "SMOOTH_S": 0.0,   # ... after averaging the probability over this many seconds
}


def _gaps(codes, cues, dead):
    """Live ball gaps of at least 1 s that start with a team in possession: (g0, g1, held, after)."""
    fps = cues["fps"]
    a, b = deadball.runs(~cues["ball_seen"] & ~dead)
    out = []
    for g0, g1 in zip(a, b):
        if g1 - g0 < fps or g0 == 0 or codes[g0 - 1] not in (HOME, AWAY):
            continue
        after = codes[g1] if g1 < len(codes) and codes[g1] in (HOME, AWAY) else None
        out.append((int(g0), int(g1), int(codes[g0 - 1]), after))
    return out


def viterbi_override(codes, cues, dead, p_home, q):
    """Rule 6 as a two-state hidden Markov model. Inside each live ball gap of at least 1 s the
    owner is the most likely sequence given the shape model: every frame adds its log-odds for
    home (scaled to per second), every change of hands costs SWITCH_COST. The gap starts with the
    team that held the ball before it and, when the touches after the gap show a team, ends with
    that team. Plainly: the ball changes hands only where the players' shape argues for it by more
    than SWITCH_COST log-odds seconds.

    With d = score(home) - score(away), the recursion is d[t] = e[t] + clip(d[t-1], -c, c), so all
    gaps are decoded together, one frame step at a time, with no loop over gaps."""
    fps, lam = cues["fps"], q["SWITCH_COST"]
    gaps = _gaps(codes, cues, dead)
    out = codes.copy()
    if not gaps:
        return out
    g0 = np.array([g[0] for g in gaps])
    ln = np.array([g[1] - g[0] for g in gaps])
    held = np.array([g[2] for g in gaps])
    after = np.array([-1 if g[3] is None else g[3] for g in gaps])
    T, G = int(ln.max()), len(gaps)
    p = np.clip(p_home, 1e-4, 1 - 1e-4)
    logit = np.log(p / (1 - p)) / fps
    t_idx = np.arange(T)
    valid = t_idx[None, :] < ln[:, None]
    frames = np.minimum(g0[:, None] + t_idx[None, :], len(codes) - 1)
    e = np.where(valid, logit[frames], 0.0)
    d = np.empty((G, T))
    prev = np.where(held == HOME, np.inf, -np.inf)          # state before the gap is fixed
    for t in range(T):
        d[:, t] = e[:, t] + np.clip(prev, -lam, lam)
        prev = d[:, t]
    last = d[np.arange(G), ln - 1]
    state = np.where(after >= 0, after, np.where(last > 0, HOME, np.where(last < 0, AWAY, held)))
    path = np.empty((G, T), np.int8)
    for t in range(T - 1, 0, -1):                           # back from each gap's last frame
        path[:, t] = state
        dp = d[:, t - 1]
        before = np.where(state == HOME, np.where(dp >= -lam, HOME, AWAY),
                          np.where(dp <= lam, AWAY, HOME))
        state = np.where(t < ln, before, state)             # frames past a gap's end: padding
    path[:, 0] = state
    for k in range(G):
        out[g0[k]:g0[k] + ln[k]] = path[k, :ln[k]]
    return out


def shape_override(codes, cues, dead, p_home, q=SHAPE):
    """Rule 6: inside each live ball gap of at least 1 s, follow the players' shape. The team
    holding the ball before the gap keeps it unless the shape model gives the other team at least
    SWITCH_P for SWITCH_S; if the touches after the gap already show a change, it is placed where
    the shape first turned (instead of GAP_FRAC)."""
    if "SWITCH_COST" in q:
        return viterbi_override(codes, cues, dead, p_home, q)
    fps = cues["fps"]
    out = codes.copy()
    a, b = deadball.runs(~cues["ball_seen"] & ~dead)
    for g0, g1 in zip(a, b):
        if g1 - g0 < fps or g0 == 0 or codes[g0 - 1] not in (HOME, AWAY):
            continue
        held = codes[g0 - 1]
        after = codes[g1] if g1 < len(codes) and codes[g1] in (HOME, AWAY) else None
        ph = p_home[g0:g1]
        if q.get("SMOOTH_S", 0) > 0:
            k = max(int(q["SMOOTH_S"] * fps), 1)
            ph = np.convolve(np.pad(ph, (k // 2, k - 1 - k // 2), mode="edge"),
                             np.ones(k) / k, mode="valid")
        p_other = ph if held == AWAY else 1 - ph
        ra, rb = deadball.runs(p_other >= q["SWITCH_P"])
        keep = (rb - ra) >= q["SWITCH_S"] * fps
        ra, rb = ra[keep], rb[keep]
        seg = np.full(g1 - g0, held, np.int8)
        for x, y in zip(ra, rb):
            seg[x:y] = 1 - held
        if after is not None and after != held:
            changed = np.flatnonzero(codes[g0:g1] != held)      # where GAP_FRAC put it
            if len(ra):
                at = ra[0]
            elif len(changed):
                at = changed[0]
            else:
                at = g1 - g0
            seg[at:] = after
        out[g0:g1] = seg
    return out


def label(cues, dead, p=PARAMS, motion=None, p_home=None, q=SHAPE):
    """Codes 0 home / 1 away / 2 dead for every frame. p_home: the shape model's probability that
    home has the ball, per frame (rule 6); without it, possession is held through ball gaps."""
    fps, n = cues["fps"], len(dead)
    tf, tt = touches(cues, p, motion)
    seen = cues["ball_seen"]
    codes = np.full(n, DEAD, np.int8)
    previous = HOME
    for sg in np.unique(cues["segment"]):
        idx = np.flatnonzero(cues["segment"] == sg)
        a, b = deadball.runs(~dead[idx])
        for s, e in zip(idx[0] + a, idx[0] + b):
            lo, hi = np.searchsorted(tf, s), np.searchsorted(tf, e)
            f, t = tf[lo:hi], tt[lo:hi]
            if not len(f):
                previous = 1 - previous                       # rule 3, no touch at all
                codes[s:e] = previous
                continue
            keep = np.r_[True, t[1:] != t[:-1]]               # changes only
            f, t = f[keep], t[keep]
            at = f.copy()
            at[0] = s                                         # rule 3: first touch owns the start
            for j in range(1, len(f)):                        # rule 4
                # the last stretch of at least 1 s without the ball between the two touches
                window = ~seen[f[j - 1]:f[j]]
                if window.any():
                    ga, gb = deadball.runs(window)
                    g0, g1 = f[j - 1] + ga[-1], f[j - 1] + gb[-1]
                    if g1 - g0 >= fps:
                        at[j] = int(g0 + p["GAP_FRAC"] * (g1 - g0))
            at = np.maximum.accumulate(at)
            for j in range(len(f)):
                stop = at[j + 1] if j + 1 < len(f) else e
                codes[at[j]:stop] = t[j]
            previous = t[-1]
    if p_home is not None:
        codes = shape_override(codes, cues, dead, p_home, q)
    return control_filter(codes, cues["segment"], fps)


def target(cues):
    """IDSSE target in the challenge's convention: DFL dead ball, DFL owner (last touch) through
    the same control filter."""
    codes = np.where(cues["dead"], DEAD, cues["owner"]).astype(np.int8)
    return control_filter(codes, cues["segment"], cues["fps"])
