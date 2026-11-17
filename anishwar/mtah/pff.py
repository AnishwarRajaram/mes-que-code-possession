"""PFF FC World Cup 2022 as a labelled test set (never for calibration). Aggregates only.

The data is request-gated under PFF's own terms and may not be redistributed. It is read in place
from the folder named by the environment variable PFF_DIR, processed on this machine only, and
everything derived from it stays in the gitignored anishwar/external/pff/. Nothing goes to any
cloud service.

Fixed before any scoring:

1. Tracking: resampled from 29.97 to 25 fps by linear interpolation on the video clock, per
   period. All 22 players, PFF's estimated positions included. The ball counts as seen only where
   PFF marks it VISIBLE on both bracketing source frames; everything else is a natural gap, so no
   ball is hidden on top. Pitch coordinates are shifted to a corner origin as in IDSSE.
2. Smoothing: label-free, the rule of smoothing.py. For the ball and for the players apart, the
   variant (PFF raw or PFF smoothed) and the moving-average K whose wobble (mean of the matches,
   live frames with the ball seen) is nearest ours in log terms.
3. Labels: specification v1 of the PFF survey. Dead from an OUT event, an END event or a challenge
   carrying a foul until the next set piece (throw-in, free kick, goal kick, corner, kick-off),
   whose team takes the ball. While alive, an event of the other team changes possession. Mapped
   to frames on the video clock, then the same 2 s control filter as the IDSSE target.
4. Scoring: the organisers' metric code on two versions with their stored constants, no refit:
   the submitted constants, read from the working tree or from a git ref.
"""

import ast
import bz2
import json
import os
import subprocess
import sys

import numpy as np

from . import calibrate, changes, config, deadball, harness, possession, shape, smoothing

PFF_DIR = os.environ.get("PFF_DIR", "")
OUT_DIR = os.path.join(config.WORKSPACE, "external", "pff")
MATCHES = ("3813", "3850", "10505", "10508", "10511", "10513", "10515", "10517")
FPS = 25
RESTARTS = {"T", "F", "G", "C", "K"}
pm = harness.pm


def _xy(o):
    if o is None or o.get("x") is None or o.get("y") is None:
        return None
    return float(o["x"]), float(o["y"])


def read(gid):
    """Source-rate arrays of one match, cached: time (ms), period, ball raw and smoothed with
    PFF visibility, players raw and smoothed (columns keyed by side and shirt number)."""
    path = os.path.join(OUT_DIR, "%s_src.npz" % gid)
    if os.path.exists(path):
        with np.load(path) as z:
            return {k: z[k] for k in z.files}
    t, per, braw, bsm, bvis = [], [], [], [], []
    cols, praw, psm = {}, [], []
    with bz2.open(os.path.join(PFF_DIR, "Tracking Data", "%s.jsonl.bz2" % gid), "rt") as f:
        for line in f:
            d = json.loads(line)
            if d.get("period") not in (1, 2, 3, 4) or d.get("videoTimeMs") is None:
                continue
            t.append(d["videoTimeMs"])
            per.append(d["period"])
            b = (d.get("balls") or [None])[0]
            xy = _xy(b)
            braw.append(xy or (np.nan, np.nan))
            bvis.append(xy is not None and b.get("visibility") == "VISIBLE")
            bs = _xy(d.get("ballsSmoothed"))
            bsm.append(bs or (np.nan, np.nan))
            rr, ss = {}, {}
            for side in ("home", "away"):
                for key, store in (("%sPlayers" % side, rr), ("%sPlayersSmoothed" % side, ss)):
                    for p in d.get(key) or []:
                        xy = _xy(p)
                        if xy is not None and p.get("jerseyNum") is not None:
                            c = (side, str(p["jerseyNum"]))
                            cols.setdefault(c, len(cols))
                            store[cols[c]] = xy
            praw.append(rr)
            psm.append(ss)
    n, k = len(t), len(cols)
    out = {"t": np.array(t, np.float64), "period": np.array(per, np.int8),
           "ball_raw": np.array(braw, np.float32), "ball_sm": np.array(bsm, np.float32),
           "ball_vis": np.array(bvis, bool),
           "team": np.array([0 if s == "home" else 1 for s, _ in sorted(cols, key=cols.get)], np.int8)}
    for name, rows in (("players_raw", praw), ("players_sm", psm)):
        a = np.full((n, k, 2), np.nan, np.float32)
        for i, r in enumerate(rows):
            for j, xy in r.items():
                a[i, j] = xy
        out[name] = a
    os.makedirs(OUT_DIR, exist_ok=True)
    np.savez_compressed(path, **out)
    return out


def _pitch(gid):
    m = json.load(open(os.path.join(PFF_DIR, "Metadata", "%s.json" % gid)))
    m = m[0] if isinstance(m, list) else m
    for p in (m.get("stadium") or {}).get("pitches") or []:
        if p.get("length") and p.get("width"):
            return float(p["length"]), float(p["width"])
    return 105.0, 68.0


def _interp(t_src, a, t):
    i1 = np.clip(np.searchsorted(t_src, t), 1, len(t_src) - 1)
    i0 = i1 - 1
    w = ((t - t_src[i0]) / np.maximum(t_src[i1] - t_src[i0], 1e-9)).astype(np.float32)
    w = w.reshape((-1,) + (1,) * (a.ndim - 1))
    return (1 - w) * a[i0] + w * a[i1], i0, i1


def frames(gid):
    """The match at 25 fps: time (ms), period, and every tracking variant, ball seen mask."""
    s = read(gid)
    length, width = _pitch(gid)
    shift = np.array([length / 2, width / 2], np.float32)
    parts = []
    for p in np.unique(s["period"]):
        idx = np.flatnonzero(s["period"] == p)
        ts = s["t"][idx]
        order = np.argsort(ts, kind="stable")
        idx, ts = idx[order], ts[order]
        keep = np.r_[True, np.diff(ts) > 0]
        idx, ts = idx[keep], ts[keep]
        grid = np.arange(ts[0], ts[-1], 1000.0 / FPS)
        part = {"t": grid, "period": np.full(len(grid), p, np.int8)}
        for name in ("ball_raw", "ball_sm", "players_raw", "players_sm"):
            part[name], i0, i1 = _interp(ts, s[name][idx], grid)
            part[name] = part[name] + shift
        vis = s["ball_vis"][idx]
        part["ball_seen"] = vis[i0] & vis[i1]
        for name in ("ball_raw", "ball_sm"):
            part[name][~part["ball_seen"]] = np.nan
        parts.append(part)
    out = {k: np.concatenate([q[k] for q in parts]) for k in parts[0]}
    out["team"], out["pitch"] = s["team"], np.array([length, width], np.float32)
    return out


def labels(gid, t_ms):
    """Specification v1 on the event stream: per frame 0 home, 1 away, 2 dead."""
    evs = json.load(open(os.path.join(PFF_DIR, "Event Data", "%s.json" % gid)))
    rows = []
    for e in evs:
        ge, pe = e.get("gameEvents") or {}, e.get("possessionEvents") or {}
        if e.get("eventTime") is None:
            continue
        rows.append((float(e["eventTime"]), ge.get("gameEventType"), str(ge.get("setpieceType")),
                     ge.get("teamId"), bool(ge.get("homeTeam")), pe.get("possessionEventType"),
                     bool((e.get("fouls") or {}).get("foulType"))))
    rows.sort(key=lambda r: r[0])
    times, states = [-1e18], [possession.DEAD]
    alive, cur = False, None
    for t, gt, sp, team, home, pet, foul in rows:
        if sp in RESTARTS:
            alive, cur = True, possession.HOME if home else possession.AWAY
            times.append(t); states.append(cur)
        elif gt in ("OUT", "END") or (foul and pet == "CH"):
            if alive:
                alive = False
                times.append(t); states.append(possession.DEAD)
        elif alive and team is not None:
            nt = possession.HOME if home else possession.AWAY
            if nt != cur:
                cur = nt
                times.append(t); states.append(cur)
    k = np.searchsorted(np.array(times), t_ms / 1000.0, side="right") - 1
    return np.array(states, np.int8)[k]


def match(gid, ball="ball_raw", players="players_raw", kb=1, kp=1):
    """Arrays in the IDSSE layout (external.load), plus the labels."""
    f = frames(gid)
    lab = labels(gid, f["t"])
    return {"ball_xy": calibrate.smooth_track(f[ball], kb), "players_xy": calibrate.smooth_track(f[players], kp),
            "team": f["team"], "period": f["period"], "pitch": f["pitch"],
            "alive": lab != possession.DEAD, "owner": np.where(lab == possession.DEAD, -1, lab).astype(np.int8)}


def fit_smoothing(matches=MATCHES):
    """smoothing.py's rule over both PFF variants. Returns the choice and the table."""
    ball, players, live, _, _ = smoothing._ours()
    target = {"ball": smoothing.wobble(ball, live), "players": smoothing.wobble(players, live)}
    fs = [frames(g) for g in matches]
    masks = []
    for f in fs:
        c = deadball.build_cues(f["ball_raw"], f["players_raw"], tuple(f["pitch"]), FPS, f["period"], f["team"])
        masks.append(~deadball.detect(c, deadball.PARAMS) & c["ball_seen"])
    chosen, table = {}, {}
    for key, variants in (("ball", ("ball_raw", "ball_sm")), ("players", ("players_raw", "players_sm"))):
        rows = []
        for v in variants:
            for k in smoothing.GRID:
                w = np.mean([smoothing.wobble(calibrate.smooth_track(f[v], k), mk) for f, mk in zip(fs, masks)], axis=0)
                rows.append(((v, k), w, float(np.sum(np.log(w / target[key]) ** 2))))
        table[key] = rows
        chosen[key] = min(rows, key=lambda r: (r[2], r[0][0] != variants[0], r[0][1]))[0]
    return {"target": target, "chosen": chosen, "table": table}


def cues(gid, choice):
    (bv, kb), (pv, kp) = choice["ball"], choice["players"]
    path = os.path.join(OUT_DIR, "%s_cues_%s%d_%s%d.npz" % (gid, bv, kb, pv, kp))
    if os.path.exists(path):
        with np.load(path) as z:
            c = {k: z[k] for k in z.files}
        c["fps"] = int(c["fps"])
        return c
    m = match(gid, bv, pv, kb, kp)
    c = deadball.build_cues(m["ball_xy"], m["players_xy"], tuple(m["pitch"]), FPS, m["period"], m["team"])
    c["dead"] = ~m["alive"]
    c["owner"] = m["owner"]
    c["shape"] = shape.team_features(m["players_xy"], m["team"], tuple(m["pitch"]), FPS, m["period"])
    np.savez_compressed(path, **c)
    return c


def sanity(choice, matches=MATCHES):
    """Label aggregates before any scoring: dead share and possessions per 90, raw and filtered."""
    rows = []
    for g in matches:
        c = cues(g, choice)
        raw = np.where(c["dead"], possession.DEAD, c["owner"]).astype(np.int8)
        tgt = possession.target(c)
        minutes = len(raw) / FPS / 60
        runs = pm.valid_runs(raw)
        rows.append((g, float(c["dead"].mean()), len(pm.possession_spells(raw, runs)) * 90 / minutes,
                     len(pm.possession_spells(tgt, pm.valid_runs(tgt))) * 90 / minutes, minutes))
    return rows


def _constants(ref=None):
    """The stored constants of a version: deadball PARAMS, possession PARAMS and SHAPE, changes MODEL
    and THRESH, shape MODEL. ref None: the working tree; otherwise read from the source at a git ref."""
    if ref is None:
        return {"dead": dict(deadball.PARAMS), "poss": dict(possession.PARAMS), "shape_q": possession.SHAPE,
                "change": changes.MODEL, "thresh": changes.THRESH, "shape": shape.MODEL}
    def assigns(rel):
        src = subprocess.run(["git", "show", "%s:anishwar/mtah/%s" % (ref, rel)], capture_output=True,
                             text=True, check=True, cwd=config.WORKSPACE).stdout
        out = {}
        for node in ast.parse(src).body:
            if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
                name = node.targets[0].id
                if name in ("PARAMS", "STOPF1", "SHAPE", "MODEL", "THRESH"):
                    try:
                        out[name] = eval(compile(ast.Expression(node.value), rel, "eval"), {"np": np, "dict": dict})
                    except NameError:
                        out[name] = eval(compile(ast.Expression(node.value), rel, "eval"),
                                         {"np": np, "dict": dict, "STOPF1": out.get("STOPF1")})
        return out
    d, p, c, s = (assigns(f) for f in ("deadball.py", "possession.py", "changes.py", "shape.py"))
    return {"dead": d["PARAMS"], "poss": p["PARAMS"], "shape_q": p["SHAPE"], "change": c["MODEL"],
            "thresh": c["THRESH"], "shape": s["MODEL"]}


def score(ref, choice, matches=MATCHES):
    """Both stages of the method on each match with the constants of ref, no refit."""
    k = _constants(ref)
    possession.PARAMS.clear()
    possession.PARAMS.update(k["poss"])
    rows = []
    for g in matches:
        v = cues(g, choice)
        gt = possession.target(v).copy()
        gt[np.flatnonzero(np.diff(v["segment"]) != 0) + 1] = harness.X
        runs = pm.valid_runs(gt)
        dead = deadball.detect(v, k["dead"])
        fr, tm, X = changes.candidates(v, possession.ball_motion(v))
        p = changes.prob(k["change"], X)
        first = changes.label(v, dead, fr, tm, p, k["thresh"])
        h = shape.prob(k["shape"], shape.inputs(v, first))
        pred = changes.label(v, dead, fr, tm, p, k["thresh"], p_home=h, q=k["shape_q"]).astype(np.int8)
        pred[gt == harness.X] = harness.X
        minutes = len(gt) / FPS / 60
        rows.append((g, pm.spell_scores(gt, pred, runs)["mean_f1"],
                     pm.macro_f1(pm.scored_confusion(pm.confusion(gt, pred))),
                     float(dead.mean()), len(pm.possession_spells(pred, runs)) * 90 / minutes))
    return rows


# Dead-ball breakdown by restart type, PFF against IDSSE (technical document Section 6.1).

SET_PIECES = {"CORNER_KICK": "C", "FREE_KICK": "F", "GOAL_KICK": "G", "KICK_OFF": "K",
              "THROW_IN": "T", "PENALTY": "P"}
RESTART_NAMES = {"C": "Corner", "T": "Throw-in", "K": "Kick-off", "F": "Free kick", "G": "Goal kick",
                 "P": "Penalty", "none": "No restart found"}


def estimated(gid):
    """Per source frame: video time (ms) and the share of players PFF marks other than VISIBLE. Cached."""
    path = os.path.join(OUT_DIR, "%s_est.npz" % gid)
    if os.path.exists(path):
        with np.load(path) as z:
            return z["t"], z["est"]
    t, e = [], []
    with bz2.open(os.path.join(PFF_DIR, "Tracking Data", "%s.jsonl.bz2" % gid), "rt") as f:
        for line in f:
            d = json.loads(line)
            if d.get("period") not in (1, 2, 3, 4) or d.get("videoTimeMs") is None:
                continue
            ps = (d.get("homePlayers") or []) + (d.get("awayPlayers") or [])
            t.append(d["videoTimeMs"])
            e.append(np.mean([p.get("visibility") != "VISIBLE" for p in ps]) if ps else np.nan)
    t, e = np.array(t, np.float64), np.array(e, np.float32)
    os.makedirs(OUT_DIR, exist_ok=True)
    np.savez_compressed(path, t=t, est=e)
    return t, e


def _grid(gid):
    """The 25 fps video times (ms) of frames(), without the interpolation."""
    s = read(gid)
    parts = []
    for p in np.unique(s["period"]):
        ts = np.unique(s["t"][s["period"] == p])
        parts.append(np.arange(ts[0], ts[-1], 1000.0 / FPS))
    return np.concatenate(parts)


def _stoppages(t, c, detected, restarts, est=None):
    """One row per true stoppage of at least 1 s: length, detected share, the first restart within
    3 s before to 5 s after its end, median tempo ratio, estimated share, missed frames by fifth."""
    a, b = deadball.runs(c["dead"])
    rt = np.array([r[0] for r in restarts])
    rows = []
    for s, f in zip(a, b):
        n = f - s
        if n < FPS:
            continue
        i = np.searchsorted(rt, t[f - 1] - 3)
        kind = restarts[i][1] if i < len(rt) and rt[i] <= t[f - 1] + 5 else "none"
        miss = ~detected[s:f]
        fifth = np.bincount(np.minimum(np.arange(n) * 5 // n, 4)[miss], minlength=5)
        rows.append((n, 1 - miss.mean(), kind, float(np.nanmedian(c["ratio"][s:f])),
                     np.nan if est is None else float(np.nanmean(est[s:f])), fifth))
    return rows


def _breakdown_pff(args):
    gid, choice, dead_k = args
    c = cues(gid, choice)
    t = _grid(gid)[:len(c["dead"])] / 1000.0
    evs = json.load(open(os.path.join(PFF_DIR, "Event Data", "%s.json" % gid)))
    restarts = sorted((float(e["eventTime"]), str((e.get("gameEvents") or {}).get("setpieceType")))
                      for e in evs if e.get("eventTime") is not None
                      and str((e.get("gameEvents") or {}).get("setpieceType")) in RESTARTS)
    ts, e = estimated(gid)
    o = np.argsort(ts, kind="stable")
    est = e[o][np.clip(np.searchsorted(ts[o] / 1000.0, t), 0, len(ts) - 1)]
    rows = _stoppages(t, c, deadball.detect(c, dead_k), restarts, est)
    return rows, float(np.nanmean(est[~c["dead"]])), float(np.nanmean(est[c["dead"]]))


def _breakdown_idsse(args):
    m, dead_k = args
    from kloppy import sportec
    from kloppy._providers.sportec import get_IDSSE_url
    from . import external
    path = lambda kind: os.path.join(external.IDSSE_DIR, get_IDSSE_url(m, kind).rsplit("/", 1)[1])
    ds = sportec.load_event(event_data=path("event"), meta_data=path("meta"))
    restarts = []
    for e in ds.events:
        sp = [q.value.name for q in (e.qualifiers or []) if type(q).__name__ == "SetPieceQualifier"]
        if sp and sp[0] in SET_PIECES:
            restarts.append((e.period.id * 1e4 + e.timestamp.total_seconds(), SET_PIECES[sp[0]]))
    restarts.sort()
    c = calibrate.idsse_cues(m)
    per = external.load(m)["period"]
    t = np.zeros(len(per))
    for q in np.unique(per):
        idx = np.flatnonzero(per == q)
        t[idx] = q * 1e4 + np.arange(len(idx)) / 25.0
    return _stoppages(t, c, deadball.detect(c, dead_k), restarts), np.nan, np.nan


def breakdown(choice, ref=None, workers=4):
    """Section 6.1 aggregates for each source under the dead-ball constants of ref."""
    from multiprocessing import Pool
    from . import external
    dead_k = _constants(ref)["dead"]
    with Pool(workers) as pool:
        res = {"IDSSE": pool.map(_breakdown_idsse, [(m, dead_k) for m in external.MATCHES]),
               "PFF": pool.map(_breakdown_pff, [(g, choice, dead_k) for g in MATCHES])}
    out = {}
    for src, parts in res.items():
        rows = [r for p in parts for r in p[0]]
        n = np.array([r[0] for r in rows], float)
        got = np.array([r[1] for r in rows])
        kind = np.array([r[2] for r in rows])
        whole, half = got == 0, got >= 0.5
        est = np.array([r[4] for r in rows])
        out[src] = {
            "stoppages": len(rows),
            "by_restart": {k: (float((n * got)[kind == k].sum() / n[kind == k].sum()), int((kind == k).sum()))
                           for k in RESTART_NAMES if (kind == k).any()},
            "missed_whole": float(n[whole].sum() / n.sum()),
            "missed_by_fifth": (np.sum([r[5] for r in rows], axis=0) / n.sum()).tolist(),
            "tempo_missed_whole": float(np.median([r[3] for r, w in zip(rows, whole) if w])),
            "tempo_caught": float(np.median([r[3] for r, h in zip(rows, half) if h])),
            "est_missed_whole": float(np.average(est[whole], weights=n[whole])),
            "est_caught": float(np.average(est[half], weights=n[half])),
            "est_live": float(np.mean([p[1] for p in parts])),
            "est_dead": float(np.mean([p[2] for p in parts])),
        }
    return out


# Stage 3 of the broadcast plan: the camera-off-the-pitch rule calibrated on PFF.

OFF_EST = 0.9                                   # off-pitch: at most about 2 of 22 players seen
CAMERA_MIN_S = (None, 2, 3, 5, 8, 12, 20)       # None: the rule is off
CAMERA_MAX_RATIO = (0.6, 0.8, 1.0, 1.2, np.inf)


def _camera_one(args):
    """Every grid point on one match: spellF1, macroF1, stoppage F1, dead share."""
    from . import broadcast
    gid, choice, k = args
    v = cues(gid, choice)
    t = _grid(gid)[:len(v["dead"])] / 1000.0
    ts, e = estimated(gid)
    o = np.argsort(ts, kind="stable")
    off = e[o][np.clip(np.searchsorted(ts[o] / 1000.0, t), 0, len(ts) - 1)] >= OFF_EST
    gt = possession.target(v).copy()
    gt[np.flatnonzero(np.diff(v["segment"]) != 0) + 1] = harness.X
    runs = pm.valid_runs(gt)
    dead0 = deadball.detect(v, k["dead"])
    fr, tm, X = changes.candidates(v, possession.ball_motion(v))
    p = changes.prob(k["change"], X)
    out, seen = {}, {}
    for n in CAMERA_MIN_S:
        for r in (CAMERA_MAX_RATIO if n is not None else (np.inf,)):
            dead = broadcast.add_offpitch(dead0, off, v["ratio"], n, r)
            key = dead.tobytes()
            if key not in seen:
                first = changes.label(v, dead, fr, tm, p, k["thresh"])
                h = shape.prob(k["shape"], shape.inputs(v, first))
                pred = changes.label(v, dead, fr, tm, p, k["thresh"], p_home=h, q=k["shape_q"]).astype(np.int8)
                pred[gt == harness.X] = harness.X
                seen[key] = (pm.spell_scores(gt, pred, runs)["mean_f1"],
                             pm.macro_f1(pm.scored_confusion(pm.confusion(gt, pred))),
                             deadball.score(dead, v["dead"], v["segment"], FPS)["f1_stop"], float(dead.mean()))
            out[(n, r)] = seen[key]
    agg = (float(off[v["dead"]].mean()), float(off[~v["dead"]].mean()))
    return gid, out, agg


def camera(choice, ref=None, workers=4):
    """Nested leave-one-match-out on PFF: the grid point with the best mean training spellF1 is
    applied to the held-out match. Returns per match (baseline, nested) scores and the choices."""
    from multiprocessing import Pool
    k = _constants(ref)
    possession.PARAMS.clear()
    possession.PARAMS.update(k["poss"])
    with Pool(workers, initializer=_set_poss, initargs=(k["poss"],)) as pool:
        res = pool.map(_camera_one, [(g, choice, k) for g in MATCHES])
    grid = list(res[0][1])
    rows = []
    for i, (g, out, agg) in enumerate(res):
        train = [res[j][1] for j in range(len(res)) if j != i]
        best = max(grid, key=lambda q: np.mean([tr[q][0] for tr in train]))
        rows.append((g, out[(None, np.inf)], out[best], best, agg))
    full = max(grid, key=lambda q: np.mean([r[1][q][0] for r in res]))
    return rows, full


def _set_poss(poss):
    possession.PARAMS.clear()
    possession.PARAMS.update(poss)


if __name__ == "__main__":
    import warnings
    warnings.simplefilter("ignore")
    if not PFF_DIR:
        sys.exit("set PFF_DIR to the PFF World Cup 2022 folder")
    step = sys.argv[1]
    if step == "read":
        for g in MATCHES:
            s = read(g)
            print("  %s: %d source frames, %d player columns" % (g, len(s["t"]), s["players_raw"].shape[1]), flush=True)
    elif step == "smoothing":
        r = fit_smoothing()
        print("ours p50/p90 (cm): ball %.2f/%.2f  players %.2f/%.2f" % (*(100 * r["target"]["ball"]), *(100 * r["target"]["players"])))
        for key in ("ball", "players"):
            for (v, kk), w, e in r["table"][key]:
                print("  %-7s %-11s K=%-2d  %.2f/%.2f cm  log error %.3f" % (key, v, kk, 100 * w[0], 100 * w[1], e))
        print("chosen:", r["chosen"])
        json.dump({k: list(v) for k, v in r["chosen"].items()}, open(os.path.join(OUT_DIR, "choice.json"), "w"))
    else:
        choice = {k: tuple(v) for k, v in json.load(open(os.path.join(OUT_DIR, "choice.json"))).items()}
        if step == "sanity":
            for g, d, raw, filt, mins in sanity(choice):
                print("  %s  %.0f min  dead %.3f  possessions/90 raw %.0f, filtered %.0f" % (g, mins, d, raw, filt))
        elif step == "score":
            for ref in (None,):
                rows = score(ref, choice)
                a = np.array([r[1:] for r in rows])
                print("%s:" % (ref or "working tree"))
                for r in rows:
                    print("  %s  spellF1 %.3f  macroF1 %.3f  dead %.3f  poss/90 %.0f" % r)
                print("  mean spellF1 %.3f  macroF1 %.3f" % (a[:, 0].mean(), a[:, 1].mean()), flush=True)
        elif step == "camera":
            rows, full = camera(choice)
            print("match   off-pitch share dead/live   spellF1 base -> nested   macroF1 base -> nested   stopF1 base -> nested   dead share   chosen (min s, max ratio)")
            for g, b, n, q, agg in rows:
                print("  %-6s %.2f / %.2f              %.3f -> %.3f          %.3f -> %.3f          %.3f -> %.3f         %.2f -> %.2f   %s"
                      % (g, agg[0], agg[1], b[0], n[0], b[1], n[1], b[2], n[2], b[3], n[3], q))
            d = np.array([n[0] - b[0] for _, b, n, _, _ in rows])
            print("spellF1 gain mean %+.4f, sd %.4f, better %d of %d, paired test %s"
                  % (d.mean(), d.std(ddof=1), (d > 0).sum(), len(d), "pass" if d.mean() > 0.93 * d.std(ddof=1) else "fail"))
            print("choice on all eight matches:", full)
        elif step == "breakdown":
            r = breakdown(choice)
            print("share of dead time detected (stoppages):   IDSSE        PFF")
            for k, name in RESTART_NAMES.items():
                cell = lambda s: "%.2f (%3d)" % r[s]["by_restart"][k] if k in r[s]["by_restart"] else "   -      "
                print("  %-28s %s   %s" % (name, cell("IDSSE"), cell("PFF")))
            for key, name, f in (("missed_whole", "dead time missed entirely", "%.3f"),
                                 ("tempo_missed_whole", "median tempo, missed entirely", "%.2f"),
                                 ("tempo_caught", "median tempo, detected >= half", "%.2f")):
                print("  %-28s %-12s %s" % (name, f % r["IDSSE"][key], f % r["PFF"][key]))
            for s in ("IDSSE", "PFF"):
                print("  %s missed by fifth of stoppage: %s" % (s, " ".join("%.3f" % x for x in r[s]["missed_by_fifth"])))
            p = r["PFF"]
            print("  PFF estimated players: missed entirely %.2f, detected >= half %.2f, live %.2f, dead %.2f"
                  % (p["est_missed_whole"], p["est_caught"], p["est_live"], p["est_dead"]))
