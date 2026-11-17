"""The broadcast recording as a camera-off-the-pitch cue on the tactical timeline.

Broadcasts cut to replays, close-ups and crowd shots mostly during stoppages, a convention of
television production rather than of any one match. Where the broadcast homography is not valid
the camera shows no usable pitch.

Stage 1, inventory(): the signal on the broadcast timeline alone.
Stage 2, align() and tactical_cue(): the broadcast is mapped to tactical frames by matching the ball
and then the players in pitch coordinates, chunk by chunk; the off-pitch cue is carried across.
All thresholds are fixed in advance, and no check here chooses a constant. Output is aggregates only.
"""

import os
import sys

import numpy as np

from . import config, deadball, load

sys.path.insert(0, os.path.join(config.REPO_ROOT, "starter"))
import load_homography  # noqa: E402  (starter)

FPS = 25
LONG_S = (3, 10, 30)
STEP = 5             # coarse search at 5 fps
CHUNK_S = 30         # broadcast chunk matched as one piece
MIN_BALL = 40        # coarse samples with the ball seen in both sources
CLIP_M = 10.0        # distances clipped so one bad sample cannot dominate
CONFIDENT_M = 2.0    # a chunk is confident when its ball error is at most this ...
SECOND_X = 2.0       # ... and the best rival offset (over 2 s away) is this many times worse
REFINE = 7           # frames searched either side of the coarse offset
SAME_CLUSTER = 25    # frames: offsets this close belong to one synchronised segment
MIN_SEGMENT = 5      # confident chunks needed to accept an offset cluster as a segment
MIN_PLAYERS = 4      # broadcast players needed to test a frame for a replay
MISMATCH_M = 3.0     # median nearest-player distance above which an on-pitch frame is a replay
OFF, UNKNOWN, ON, REPLAY = 2, 0, 1, 3
CAMERA_MIN_S = 8         # stage 3: chosen in every fold of the nested PFF test (pff.camera)
CAMERA_MAX_RATIO = np.inf  # no tempo condition, likewise chosen in every fold


def frames():
    """Per broadcast frame: homography valid, reprojection error, players and ball visible."""
    _, err, valid = load_homography.load(source="broadcast")
    atd = load.load_atd("broadcast")
    players = np.isfinite(atd["players"][:, :, 0]).sum(axis=1)
    ball = np.isfinite(atd["ball"][:, 0])
    n = min(len(valid), len(players))
    return {"valid": valid[:n], "err": err[:n], "players": players[:n], "ball": ball[:n]}


def inventory():
    """Aggregates of the off-pitch stretches and of what the tracking sees on and off the pitch."""
    f = frames()
    off = ~f["valid"]
    a, b = deadball.runs(off)
    sec = (b - a) / FPS
    minutes = len(off) / FPS / 60
    tac = load.load_atd("tactical")
    tac_players = np.isfinite(tac["players"][:, :, 0]).sum(axis=1)
    on = ~off
    return {
        "frames": len(off), "minutes": minutes, "tactical_frames": len(tac_players),
        "valid_share": float(on.mean()),
        "stretches": len(sec),
        "stretch_s": {"median": float(np.median(sec)), "p90": float(np.percentile(sec, 90)),
                      "max": float(sec.max())},
        "stretches_at_least": {s: int((sec >= s).sum()) for s in LONG_S},
        "off_time_in_stretches_at_least": {s: float(sec[sec >= s].sum() / sec.sum()) for s in LONG_S},
        "players_median": {"valid": float(np.median(f["players"][on])),
                           "not_valid": float(np.median(f["players"][off])),
                           "tactical": float(np.median(tac_players))},
        "ball_seen": {"valid": float(f["ball"][on].mean()), "not_valid": float(f["ball"][off].mean())},
        "err_valid": {"median": float(np.nanmedian(f["err"][on])),
                      "p90": float(np.nanpercentile(f["err"][on], 90))},
    }


def _tracks(source):
    atd = load.load_atd(source)
    on = (atd["team"] == load.HOME) | (atd["team"] == load.AWAY)
    return load.to_metres(atd["ball"][:, :2]), load.to_metres(atd["players"][:, on, :2])


def _coarse(bb, tb, n):
    """Best tactical sample for a broadcast chunk of n samples: (index, error m, rival error m)."""
    from numpy.lib.stride_tricks import sliding_window_view
    m = np.isfinite(bb[:, 0])
    X = sliding_window_view(tb[:, 0], n)[:, m]
    Y = sliding_window_view(tb[:, 1], n)[:, m]
    d = np.minimum(np.hypot(X - bb[m, 0], Y - bb[m, 1]), CLIP_M)
    cnt = np.isfinite(d).sum(axis=1)
    err = np.where(cnt >= MIN_BALL / 2, np.nansum(d, axis=1) / np.maximum(cnt, 1), np.inf)
    k = int(np.argmin(err))
    far = np.abs(np.arange(len(err)) - k) > 2 * FPS // STEP
    return k, float(err[k]), float(err[far].min())


def _chamfer(bp, tp, bb, tb):
    """Mean over frames of the median distance from each broadcast player to the nearest tactical
    player, and the mean ball distance; both clipped."""
    d = np.linalg.norm(bp[:, :, None, :] - tp[:, None, :, :], axis=-1)
    near = np.nanmin(np.where(np.isfinite(d), d, np.inf), axis=2)
    near[~np.isfinite(near)] = np.nan
    pl = np.nanmean(np.minimum(np.nanmedian(near, axis=1), CLIP_M))
    ball = np.nanmean(np.minimum(np.hypot(*(bb - tb).T), CLIP_M))
    return float(pl), float(ball)


def align(mirror=False):
    """Per broadcast chunk: start frame, offset in frames (tactical = broadcast + offset), coarse
    ball error, rival error, refined player and ball errors, confident flag, valid share."""
    f = frames()
    bb, bp = _tracks("broadcast")
    tb, tp = _tracks("tactical")
    n = len(f["valid"])
    bb, bp = bb[:n].copy(), bp[:n].copy()
    if mirror:
        size = np.asarray(config.PITCH, np.float32)
        bb, bp = size - bb, size - bp
    bb[~f["valid"]] = np.nan
    bp[~f["valid"]] = np.nan
    cb, ct = bb[::STEP], tb[::STEP]
    L = CHUNK_S * FPS // STEP
    rows = []
    for c in range(0, len(cb) - L + 1, L):
        start, valid = c * STEP, float(f["valid"][c * STEP:(c + L) * STEP].mean())
        if np.isfinite(cb[c:c + L, 0]).sum() < MIN_BALL:
            rows.append((start, None, np.nan, np.nan, np.nan, np.nan, False, valid))
            continue
        k, e, rival = _coarse(cb[c:c + L], ct, L)
        base = (k - c) * STEP
        fr = np.arange(start, start + L * STEP, STEP)
        best = None
        for o in range(base - REFINE, base + REFINE + 1):
            j = fr + o
            ok = (j >= 0) & (j < len(tb))
            if ok.sum() < L // 2:
                continue
            pl, ball = _chamfer(bp[fr[ok]], tp[j[ok]], bb[fr[ok]], tb[j[ok]])
            score = pl + ball
            if best is None or score < best[0]:
                best = (score, o, pl, ball)
        conf = e <= CONFIDENT_M and rival >= SECOND_X * e
        rows.append((start, best[1], e, rival, best[2], best[3], conf, valid))
    return rows


def _clusters(rows):
    offs = sorted(r[1] for r in rows if r[6])
    groups = []
    for o in offs:
        if groups and o - groups[-1][-1] <= SAME_CLUSTER:
            groups[-1].append(o)
        else:
            groups.append([o])
    return groups


def segments(rows):
    """Synchronised segments: each offset cluster of at least MIN_SEGMENT confident chunks, with a
    linear drift fit over broadcast frames. [(first broadcast frame, last, slope, intercept)]."""
    L = CHUNK_S * FPS
    out = []
    for g in _clusters(rows):
        if len(g) < MIN_SEGMENT:
            continue
        rs = [r for r in rows if r[6] and g[0] <= r[1] <= g[-1]]
        x = np.array([r[0] + L / 2 for r in rs])
        y = np.array([r[1] for r in rs], float)
        slope, icept = np.polyfit(x, y, 1)
        out.append((min(r[0] for r in rs) - L, max(r[0] for r in rs) + 2 * L, slope, icept))
    return out


def tactical_cue(rows):
    """Per tactical frame: UNKNOWN, ON, OFF, or REPLAY. Inside a segment every broadcast frame is
    carried to tactical by the segment's fit; an on-pitch frame whose players sit over MISMATCH_M
    (median nearest distance) from the tactical players at that moment is a replay."""
    f = frames()
    _, bp = _tracks("broadcast")
    _, tp = _tracks("tactical")
    n_b, n_tac = len(f["valid"]), config.N_FRAMES
    state = np.zeros(n_tac, np.int8)
    for lo, hi, slope, icept in segments(rows):
        b = np.arange(max(lo, 0), min(hi, n_b))
        j = np.rint(b + slope * b + icept).astype(int)
        ok = (j >= 0) & (j < n_tac)
        b, j = b[ok], j[ok]
        v = f["valid"][b]
        st = np.where(v, ON, OFF).astype(np.int8)
        for k in range(0, len(b), 5000):
            sl = slice(k, k + 5000)
            vb = v[sl]
            d = np.linalg.norm(bp[b[sl]][vb][:, :, None, :] - tp[j[sl]][vb][:, None, :, :], axis=-1)
            near = np.nanmin(np.where(np.isfinite(d), d, np.inf), axis=2)
            near[~np.isfinite(near)] = np.nan
            cnt = np.isfinite(near).sum(axis=1)
            med = np.nanmedian(np.where(cnt[:, None] >= MIN_PLAYERS, near, np.nan), axis=1)
            idx = np.flatnonzero(vb) + k
            st[idx[med > MISMATCH_M]] = REPLAY
        state[j] = st
    return state


def add_offpitch(dead, off, ratio, min_s, max_ratio, fps=FPS):
    """Stage 3 rule: every off-pitch stretch of at least min_s seconds whose median player tempo
    ratio is at most max_ratio is dead. The rule only adds dead ball; min_s None leaves dead as is."""
    out = dead.copy()
    if min_s is None:
        return out
    a, b = deadball.runs(off)
    for s, e in zip(a, b):
        if e - s >= min_s * fps and np.nanmedian(ratio[s:e]) <= max_ratio:
            out[s:e] = True
    return out


def checks(state):
    """Label-free agreement of the carried cue with the submitted dead-ball detector and the tagging."""
    from . import calibrate, census
    cues, graded = calibrate.our_cues()
    dead = deadball.detect(cues)
    g = graded
    cov = state != UNKNOWN
    off, on = (state == OFF) & g, (state == ON) & g
    a, b = deadball.runs(off)
    long_ = [(s, e) for s, e in zip(a, b) if e - s >= 3 * FPS]
    tag = [e["start_frame"] for e in load.load_tagging()
           if "set piece" in (e["code"] or "").lower() and 0 <= e["start_frame"] < len(dead) and g[e["start_frame"]]]
    before = sum(bool(off[max(0, s - 30 * FPS):s].any()) for s in tag)
    at_on = sum(bool(state[s] == ON) for s in tag)
    return {"coverage": float(cov[g].mean()), "off_share": float(off.sum() / max((cov & g).sum(), 1)),
            "replay_share": float(((state == REPLAY) & g).sum() / max((cov & g).sum(), 1)),
            "dead_given_off": float(dead[off].mean()), "dead_given_on": float(dead[on].mean()),
            "off_given_dead": float(off[dead & cov].mean()),
            "off_runs_3s": len(long_), "off_runs_mostly_dead": sum(dead[s:e].mean() >= 0.5 for s, e in long_),
            "set_pieces": len(tag), "off_in_30s_before": before, "on_at_start": at_on}


def stage2():
    for mirror in (False, True):
        rows = align(mirror)
        ok = [r for r in rows if r[1] is not None]
        conf = [r for r in ok if r[6]]
        print("mirror %s: chunks %d, with ball %d, confident %d, median ball error %.2f m"
              % (mirror, len(rows), len(ok), len(conf), np.median([r[2] for r in ok]) if ok else np.nan))
        if mirror is False:
            keep = rows
    rows = keep
    conf = [r for r in rows if r[6]]
    print("confident chunks: coarse ball %.2f m, refined players %.2f m, refined ball %.2f m (medians)"
          % tuple(np.median([r[k] for r in conf]) for k in (2, 4, 5)))
    for g in _clusters(rows):
        span = [r[0] / FPS / 60 for r in conf if g[0] <= r[1] <= g[-1]]
        print("  offset %+.1f s (spread %d frames): %d chunks, broadcast min %.0f-%.0f"
              % (np.median(g) / FPS, g[-1] - g[0], len(g), min(span), max(span)))
    for lo, hi, slope, icept in segments(rows):
        print("  segment broadcast min %.0f-%.0f: drift %.2f s per hour" % (lo / FPS / 60, hi / FPS / 60, slope * 3600))
    state = tactical_cue(rows)
    r = checks(state)
    print("graded tactical frames covered %.3f; of covered: off-pitch %.3f, replay suspect %.3f"
          % (r["coverage"], r["off_share"], r["replay_share"]))
    print("detected dead share: off-pitch %.2f, on-pitch %.2f; off-pitch share of detected dead %.2f"
          % (r["dead_given_off"], r["dead_given_on"], r["off_given_dead"]))
    print("off-pitch runs >= 3 s: %d, mostly inside detected dead %d" % (r["off_runs_3s"], r["off_runs_mostly_dead"]))
    print("tagged set pieces %d: off-pitch in the 30 s before %d, on-pitch at start %d"
          % (r["set_pieces"], r["off_in_30s_before"], r["on_at_start"]))


def apply(dead, cues):
    """The submission's use of the cue: the off-pitch rule on top of the dead-ball rules."""
    return add_offpitch(dead, tactical_cue(align()) == OFF, cues["ratio"], CAMERA_MIN_S, CAMERA_MAX_RATIO)


if __name__ == "__main__" and sys.argv[1:] == ["align"]:
    stage2()
elif __name__ == "__main__":
    r = inventory()
    print("broadcast %d frames (%.0f min), tactical %d frames" % (r["frames"], r["minutes"], r["tactical_frames"]))
    print("homography valid %.3f; off-pitch stretches %d, median %.1f s, p90 %.1f s, max %.1f s"
          % (r["valid_share"], r["stretches"], *r["stretch_s"].values()))
    for s in LONG_S:
        print("  stretches >= %2d s: %3d, holding %.2f of off-pitch time"
              % (s, r["stretches_at_least"][s], r["off_time_in_stretches_at_least"][s]))
    p, bl, e = r["players_median"], r["ball_seen"], r["err_valid"]
    print("median players tracked: valid %.0f, not valid %.0f (tactical %.0f)" % (p["valid"], p["not_valid"], p["tactical"]))
    print("ball tracked: valid %.2f, not valid %.2f" % (bl["valid"], bl["not_valid"]))
    print("reprojection error where valid: median %.2f, p90 %.2f" % (e["median"], e["p90"]))
