"""Which tracking cues separate dead ball from live play, measured on other matches (IDSSE).

Labels are the DFL's own per-frame BallStatus and BallPossession on seven Bundesliga matches, so
this is allowed calibration data (RULES.md: labels on other games). Output is aggregate only:

  rhythm     dead share, stoppages and possessions per 90 minutes, length percentiles
  cues       per-frame separation of dead vs alive (AUC: 0.5 = useless, 1.0 = perfect) for
             player speed (absolute and relative to the local norm), ball speed, ball out of
             the pitch, nearest player to the ball, share of players standing, kick-off shape
  timing     around each dead stretch: how far the ball-out, player slow-down, restart strike
             and players' speed-up sit from the DFL's own dead-ball start and end
  restarts   does the team in possession change across a stoppage?
"""

import numpy as np
import pandas as pd

from . import external

FPS = 25
NORM_WINDOW_S = 300
SMOOTH_S = 1.0


def _runs(mask):
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def _speed(xy):
    """Central-difference speed in m/s along axis 0 (works for (n,2) and (n,p,2))."""
    v = np.full(xy.shape[:-1], np.nan, np.float32)
    d = (xy[2:] - xy[:-2]) * (FPS / 2.0)
    v[1:-1] = np.sqrt((d ** 2).sum(-1))
    return v


def features(m):
    """Per-frame cue arrays for one match (dict from external.load)."""
    length, width = m["pitch"]
    ball, pxy = m["ball_xy"], m["players_xy"]
    with np.errstate(invalid="ignore"):
        pspeed = _speed(pxy)
        mean_speed = np.nanmean(np.where(pspeed < 12, pspeed, np.nan), axis=1)
        smooth = pd.Series(mean_speed).rolling(int(SMOOTH_S * FPS), center=True, min_periods=1).mean()
        per_s = smooth.groupby(np.arange(len(smooth)) // FPS).mean()
        norm = per_s.rolling(NORM_WINDOW_S, center=True, min_periods=30).median()
        norm = np.repeat(norm.values, FPS)[:len(smooth)]
        ratio = smooth.values / norm
        bspeed = _speed(ball)
        x, y = ball[:, 0], ball[:, 1]
        beyond = np.maximum.reduce([-x, x - length, -y, y - width])     # metres past a line
        d = np.sqrt(((pxy - ball[:, None, :]) ** 2).sum(-1))
        nearest = np.nanmin(d, axis=1)
        standing = np.nanmean(np.where(np.isnan(pspeed), np.nan, pspeed < 1.0), axis=1)
        half = pxy[:, :, 0] < length / 2
        present = ~np.isnan(pxy[:, :, 0])
        shares = []
        for t in (0, 1):
            on = present & (m["team"] == t)[None, :]
            shares.append((half & on).sum(1) / np.maximum(on.sum(1), 1))
        kickoff = np.minimum(np.maximum(shares[0], 1 - shares[0]), np.maximum(shares[1], 1 - shares[1]))
        opposite = np.abs(shares[0] - shares[1])
        kickoff = np.where(opposite > 0.5, kickoff, 0.0)
    return {"speed": smooth.values, "ratio": ratio, "ball_speed": bspeed, "beyond": beyond,
            "nearest": nearest, "standing": standing, "kickoff": kickoff}


def _auc(score, dead):
    """P(score on a dead frame > score on an alive frame); ties count half."""
    ok = ~np.isnan(score)
    s, y = score[ok], dead[ok]
    ranks = pd.Series(s).rank().values
    n1, n0 = y.sum(), (~y).sum()
    return float((ranks[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else float("nan")


def first_true(mask, lo, hi, reverse=False):
    lo, hi = max(lo, 0), min(hi, len(mask))
    idx = np.flatnonzero(mask[lo:hi])
    if not len(idx):
        return None
    return lo + int(idx[-1] if reverse else idx[0])


def analyse(match_ids=external.MATCHES):
    pooled = {k: [] for k in ("speed", "ratio", "ball_speed", "beyond", "nearest", "standing",
                              "kickoff", "dead")}
    rhythm = {"minutes": 0.0, "dead_frames": 0, "stops": [], "poss": [], "flicker": 0,
              "same": 0, "changed": 0}
    timing = {"out_vs_start": [], "slow_vs_start": [], "strike_vs_end": [], "resume_vs_end": [],
              "stops_with_ball_out": 0, "stops": 0}
    for mid in match_ids:
        m = external.load(mid)
        f = features(m)
        dead = ~m["alive"]
        for k in pooled:
            pooled[k].append(dead if k == "dead" else f[k])
        rhythm["minutes"] += len(dead) / FPS / 60
        rhythm["dead_frames"] += int(dead.sum())

        # Possessions: maximal runs of one owner while alive, within a period.
        for p in (1, 2):
            in_p = m["period"] == p
            seg_codes = np.where(dead, 2, m["owner"])[in_p]
            bounds = np.flatnonzero(np.diff(seg_codes) != 0) + 1
            starts = np.concatenate(([0], bounds))
            stops_ = np.concatenate((bounds, [len(seg_codes)]))
            segs = [(a, b, int(seg_codes[a])) for a, b in zip(starts, stops_)]
            for i, (a, b, k) in enumerate(segs):
                if k in (0, 1):
                    rhythm["poss"].append((b - a) / FPS)
                elif k == 2:
                    rhythm["stops"].append((b - a) / FPS)
                if i + 2 < len(segs):
                    k1, k2 = segs[i + 1][2], segs[i + 2][2]
                    if k in (0, 1) and k2 == k and k1 in (0, 1) and k1 != k \
                            and (segs[i + 1][1] - segs[i + 1][0]) / FPS < 2:
                        rhythm["flicker"] += 1
                    if k in (0, 1) and k1 == 2 and k2 in (0, 1):
                        rhythm["same" if k2 == k else "changed"] += 1

        out = f["beyond"] > 0
        slow = f["ratio"] < 0.65
        fast = f["ratio"] > 0.75
        strike = np.zeros(len(dead), bool)
        bs = np.nan_to_num(f["ball_speed"])
        strike[5:] = (bs[5:] >= 4.0) & (bs[4:-1] < 4.0) & (bs[:-5] < 1.5)
        for a, b in _runs(dead):
            if b - a < FPS:                      # sub-second flags are not stoppages
                continue
            timing["stops"] += 1
            o = first_true(out, a - 5 * FPS, a + 5 * FPS)
            if o is not None:
                timing["stops_with_ball_out"] += 1
                timing["out_vs_start"].append((o - a) / FPS)
            s = first_true(slow, a - 5 * FPS, b)
            if s is not None:
                timing["slow_vs_start"].append((s - a) / FPS)
            k = first_true(strike, b - 5 * FPS, b + 3 * FPS)
            if k is not None:
                timing["strike_vs_end"].append((k - b) / FPS)
            r = first_true(fast, b - 5 * FPS, b + 10 * FPS)
            if r is not None:
                timing["resume_vs_end"].append((r - b) / FPS)

    pooled = {k: np.concatenate(v) for k, v in pooled.items()}
    dead = pooled.pop("dead")
    aucs = {}
    for k, v in pooled.items():
        a = _auc(v, dead)
        aucs[k] = (a, "higher when dead" if a >= 0.5 else "lower when dead")
    return rhythm, timing, aucs, {k: (np.nanmedian(v[dead]), np.nanmedian(v[~dead]))
                                  for k, v in pooled.items()}


def report(match_ids=external.MATCHES):
    rhythm, timing, aucs, medians = analyse(match_ids)
    pct = lambda v: " ".join("%+.1f" % np.percentile(v, q) for q in (10, 25, 50, 75, 90)) \
        if len(v) else "n=0"  # noqa: E731
    per90 = 90.0 / rhythm["minutes"]
    stops, poss = np.array(rhythm["stops"]), np.array(rhythm["poss"])
    lines = [
        "IDSSE, %d matches, %.0f minutes of play" % (len(match_ids), rhythm["minutes"]),
        "",
        "rhythm (DFL labels)",
        "  dead share %.1f%%   stoppages %.0f per 90 (length s p10/25/50/75/90: %s)" % (
            100 * rhythm["dead_frames"] / (rhythm["minutes"] * 60 * FPS), len(stops) * per90,
            pct(stops)),
        "  stoppages under 1 s %.0f%%, under 5 s %.0f%%" % (100 * (stops < 1).mean(),
                                                         100 * (stops < 5).mean()),
        "  possessions %.0f per 90 (length s p10/25/50/75/90: %s), under 2 s %.0f%%" % (
            len(poss) * per90, pct(poss), 100 * (poss < 2).mean()),
        "  quick returns %.1f per 100 possessions;  team changes across a stoppage %d / same %d" % (
            100 * rhythm["flicker"] / max(len(poss), 1), rhythm["changed"], rhythm["same"]),
        "",
        "cues, dead vs alive per frame: AUC (median dead / median alive)",
    ]
    names = {"speed": "players' mean speed m/s", "ratio": "speed / local 5-min norm",
             "ball_speed": "ball speed m/s", "beyond": "ball metres past a line",
             "nearest": "nearest player to ball m", "standing": "share of players < 1 m/s",
             "kickoff": "kick-off shape score"}
    for k, (a, how) in sorted(aucs.items(), key=lambda kv: -abs(kv[1][0] - 0.5)):
        lines.append("  %-28s AUC %.2f  %-17s (%.2f / %.2f)" % (
            names[k], max(a, 1 - a), how, medians[k][0], medians[k][1]))
    lines += [
        "",
        "timing around the DFL's own stoppages (%d of at least 1 s), seconds" % timing["stops"],
        "  ball past a line within 5 s of the start: %.0f%% of stoppages" % (
            100 * timing["stops_with_ball_out"] / max(timing["stops"], 1)),
        "  ball crosses the line, relative to dead start  p10/25/50/75/90: %s" % pct(timing["out_vs_start"]),
        "  players drop below 65%% of norm, vs dead start: %s  (n=%d)" % (
            pct(timing["slow_vs_start"]), len(timing["slow_vs_start"])),
        "  restart strike (still, then > 4 m/s), vs dead end: %s  (n=%d)" % (
            pct(timing["strike_vs_end"]), len(timing["strike_vs_end"])),
        "  players back above 75%% of norm, vs dead end: %s  (n=%d)" % (
            pct(timing["resume_vs_end"]), len(timing["resume_vs_end"])),
    ]
    return "\n".join(lines)
