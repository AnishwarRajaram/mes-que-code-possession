"""How much to smooth IDSSE's tracking so that it wobbles like ours. Label-free; aggregates only.

Our automatic tracking is smoother than IDSSE's. The rules and models are calibrated on IDSSE, so
IDSSE is smoothed first (calibrate.SMOOTH_IDSSE, calibrate.smooth_track: a centred moving average
of K frames applied twice). K is chosen here, for the ball and for the players apart, on one
measure only: the frame-to-frame wobble, the second difference |x[t+1] - 2x[t] + x[t-1]|, on live
frames with the ball seen (our dead-ball rules on each source), at its 50th and 90th percentiles.
The K whose IDSSE wobble (mean of the seven matches) is nearest ours in log terms is chosen.

Then measures that were NOT fitted are compared, with IDSSE's ball hidden like ours as in
calibration: how near the nearest player is to the ball, how often the ball is slow, and how many
change candidates (changes.candidates) a live minute has. They say whether the smoothing makes
IDSSE look like ours beyond the measure it was fitted on.
"""

import numpy as np

from . import calibrate, changes, deadball, external, load, possession, shape

GRID = (1, 3, 5, 7, 9, 11, 13, 15, 19, 25)
UNFITTED = ("nearest player, any ball (m)", "nearest player, slow ball (m)", "slow ball share %",
            "candidates per live minute", "candidate turn p50 (deg)",
            "candidate |speed change| p75")


def second_diff(xy):
    d = np.full(xy.shape[:-1], np.nan)
    d[1:-1] = np.linalg.norm(xy[2:] - 2 * xy[1:-1] + xy[:-2], axis=-1)
    return d


def wobble(xy, mask):
    return np.nanpercentile(second_diff(xy)[mask], (50, 90))


def _ours():
    atd = load.load_atd("tactical")
    on = (atd["team"] == load.HOME) | (atd["team"] == load.AWAY)
    c, graded = calibrate.our_cues()
    live = ~deadball.detect(c) & c["ball_seen"]
    return (load.to_metres(atd["ball"][:, :2]), load.to_metres(atd["players"][:, on, :2]), live,
            c, graded)


def _raw_cues(m):
    c = deadball.build_cues(m["ball_xy"], m["players_xy"], tuple(m["pitch"]), 25, m["period"],
                            m["team"])
    c["shape"] = shape.team_features(m["players_xy"], m["team"], tuple(m["pitch"]), 25, m["period"])
    c["dead"] = ~m["alive"]
    return c


def _unfitted(c, live):
    motion = possession.ball_motion(c)
    near = np.fmin(c["near_home"], c["near_away"])
    slow = live & (np.nan_to_num(motion[0], nan=99) < 2.0)
    fr, _, X = changes.candidates(c, motion)
    dead = deadball.detect(c)
    keep = ~dead[fr]
    return np.r_[np.nanmedian(near[live]), np.nanmedian(near[slow]),
                 100 * slow.sum() / max(live.sum(), 1),
                 keep.sum() / ((~dead).sum() / c["fps"] / 60),
                 np.median(X[keep][:, 0]), np.percentile(np.abs(X[keep][:, 1]), 75)]


def run():
    ball, players, live, c_ours, graded = _ours()
    target = {"ball_xy": wobble(ball, live), "players_xy": wobble(players, live)}
    ms = [external.load(mid) for mid in external.MATCHES]
    masks = []
    for m in ms:
        c = _raw_cues(m)
        masks.append(~deadball.detect(c) & c["ball_seen"])
    table, chosen = {}, {}
    for key in ("ball_xy", "players_xy"):
        rows = []
        for k in GRID:
            w = np.mean([wobble(calibrate.smooth_track(m[key], k), mk)
                         for m, mk in zip(ms, masks)], axis=0)
            rows.append((k, w, float(np.sum(np.log(w / target[key]) ** 2))))
        table[key] = rows
        chosen[key] = min(rows, key=lambda r: (r[2], r[0]))[0]

    profile = deadball.ball_gap_profile(c_ours, graded)
    unfitted = {"raw": [], "smoothed": []}
    for i, m in enumerate(ms):
        for tag, kb, kp in (("raw", 1, 1), ("smoothed", chosen["ball_xy"], chosen["players_xy"])):
            sm = dict(m, ball_xy=calibrate.smooth_track(m["ball_xy"], kb),
                      players_xy=calibrate.smooth_track(m["players_xy"], kp))
            v = deadball.dropout(_raw_cues(sm), profile, seed=100 * i)
            unfitted[tag].append(_unfitted(v, ~deadball.detect(v) & v["ball_seen"]))
    return {"target": target, "table": table, "chosen": chosen,
            "unfitted": {"ours": _unfitted(c_ours, live),
                         "raw": np.mean(unfitted["raw"], axis=0),
                         "smoothed": np.mean(unfitted["smoothed"], axis=0)}}


def report(r=None):
    r = run() if r is None else r
    lines = ["smoothing IDSSE to wobble like our tracking (live frames, ball seen; label-free)"]
    for key, name in (("ball_xy", "ball"), ("players_xy", "players")):
        t = r["target"][key]
        lines += ["", "%s: ours p50 / p90 %.2f / %.2f cm" % (name, 100 * t[0], 100 * t[1]),
                  "  K    IDSSE p50 / p90 (cm)   log error"]
        for k, w, err in r["table"][key]:
            lines.append("  %-3d  %6.2f / %5.2f         %.3f%s" % (
                k, 100 * w[0], 100 * w[1], err, "   <- chosen" if k == r["chosen"][key] else ""))
    lines += ["", "not fitted (IDSSE ball hidden like ours):",
              "  %-32s %8s %8s %10s" % ("", "ours", "IDSSE", "smoothed")]
    u = r["unfitted"]
    for j, name in enumerate(UNFITTED):
        lines.append("  %-32s %8.2f %8.2f %10.2f" % (name, u["ours"][j], u["raw"][j],
                                                      u["smoothed"][j]))
    lines += ["", "stored: calibrate.BALL_SMOOTH_K %d, PLAYER_SMOOTH_K %d -> %s" % (
        calibrate.BALL_SMOOTH_K, calibrate.PLAYER_SMOOTH_K,
        "matches" if (r["chosen"]["ball_xy"], r["chosen"]["players_xy"])
        == (calibrate.BALL_SMOOTH_K, calibrate.PLAYER_SMOOTH_K) else "DIFFERS")]
    return "\n".join(lines)
