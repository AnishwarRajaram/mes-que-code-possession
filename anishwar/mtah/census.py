"""Sequence census: what patterns does a set of labels produce? Aggregates only.

Checks a labelling for plausibility without any ground truth, by counting the sequences it makes
and comparing them with published football norms and with data from other matches:

  time       match minutes, dead and live ("effective") minutes,
             dead share overall, per half and per 15-minute block.
  possessions  count, rate, length percentiles, share under 1 s and 2 s.
  stoppages    count, rate, length percentiles, share under 5 s and over 60 s.
  flicker      a possession under FLICKER_S sandwiched between two of the other team (H A H or
               A H A): either a real quick turnover or the labeller hesitating.
  restarts     across each stoppage (H D A vs H D H): does the team change? In football the
               restart usually goes to the side that did not put the ball out, or was fouled.
  live spells  live stretches between stoppages: length percentiles and possessions in each.

`detector_breakdown` adds, for the dead-ball detector's own stoppages, how many came from each
rule and the dead minutes each rule contributes.

Nothing here prints a frame or a label; every output is a count, a share or a percentile.
"""

import numpy as np

from . import config
from .harness import pm, H, A, D, X

FLICKER_S = 2.0
BLOCK_MIN = 15
PERCENTILES = (10, 25, 50, 75, 90)


def _pct(values):
    if not len(values):
        return {p: 0.0 for p in PERCENTILES}
    return {p: float(np.percentile(values, p)) for p in PERCENTILES}


def _runs_for(codes, runs=None):
    """Graded spans: the given runs, else the two halves."""
    if runs is not None:
        return runs
    return [(first, last + 1) for _, first, last in config.half_blocks()]


def census(codes, runs=None, exclude=None):
    """Aggregates for one labelling. runs: graded spans (default both halves). exclude: an
    optional (start, stop) span left out of the time accounting."""
    fps = float(config.FPS)
    runs = _runs_for(codes, runs)

    timed = np.zeros(len(codes), bool)
    for a, b in runs:
        timed[a:b] = True
    if exclude:
        timed[exclude[0]:exclude[1]] = False
    match_min = timed.sum() / fps / 60
    dead_min = (timed & (codes == D)).sum() / fps / 60

    blocks = []
    for half, (a, b) in enumerate(runs, 1):
        if exclude and a <= exclude[0] < b:
            a = max(a, exclude[1])
        step = int(BLOCK_MIN * 60 * fps)
        for i, lo in enumerate(range(a, b, step)):
            hi = min(b, lo + step)
            if hi - lo < 60 * fps:
                continue
            blocks.append(("H{} {}-{}'".format(half, i * BLOCK_MIN, i * BLOCK_MIN + BLOCK_MIN),
                           float((codes[lo:hi] == D).mean())))
    halves = []
    for a, b in runs:
        if exclude and a <= exclude[0] < b:
            a = max(a, exclude[1])
        halves.append(float((codes[a:b] == D).mean()))

    segs = pm.segments(codes, runs)
    poss = np.array([b - a for a, b, k in segs if k in (H, A)]) / fps
    stops = np.array([b - a for a, b, k in segs if k == D and not (exclude and a == exclude[0])]) / fps

    flicker = same_team = changed = 0
    live_lengths, live_poss = [], []
    current_live, current_n = 0, 0
    for i, (a, b, k) in enumerate(segs):
        if i + 2 < len(segs):
            (a1, b1, k1), (a2, b2, k2) = segs[i + 1], segs[i + 2]
            same_run = any(lo <= a and b2 <= hi for lo, hi in runs)
            if same_run and k in (H, A) and k2 == k and k1 in (H, A) and k1 != k \
                    and b1 == a2 and b == a1 and (b1 - a1) / fps < FLICKER_S:
                flicker += 1
            if same_run and k in (H, A) and k1 == D and k2 in (H, A) and b == a1 and b1 == a2:
                if k2 == k:
                    same_team += 1
                else:
                    changed += 1
        if k in (H, A):
            current_live += b - a
            current_n += 1
        nxt = segs[i + 1] if i + 1 < len(segs) else None
        if k in (H, A) and (nxt is None or nxt[2] not in (H, A) or nxt[0] != b):
            live_lengths.append(current_live / fps)
            live_poss.append(current_n)
            current_live, current_n = 0, 0

    return {
        "match_min": match_min,
        "dead_min": dead_min,
        "live_min": match_min - dead_min,
        "dead_share": dead_min / match_min if match_min else 0.0,
        "dead_share_halves": halves,
        "dead_share_blocks": blocks,
        "n_poss": len(poss),
        "poss_per_min": len(poss) / match_min if match_min else 0.0,
        "poss_pct": _pct(poss),
        "poss_under_1s": float((poss < 1).mean()) if len(poss) else 0.0,
        "poss_under_2s": float((poss < 2).mean()) if len(poss) else 0.0,
        "n_stops": len(stops),
        "stops_per_min": len(stops) / match_min if match_min else 0.0,
        "stop_pct": _pct(stops),
        "stops_under_5s": float((stops < 5).mean()) if len(stops) else 0.0,
        "stops_over_60s": float((stops > 60).mean()) if len(stops) else 0.0,
        "flicker": flicker,
        "flicker_per_100_poss": 100.0 * flicker / len(poss) if len(poss) else 0.0,
        "restart_same": same_team,
        "restart_changed": changed,
        "live_pct": _pct(np.array(live_lengths)),
        "poss_per_live_pct": _pct(np.array(live_poss)),
    }


SET_PIECE_BEFORE_S = 5.0  # a tagged set piece is "caught" if any dead frame lies within this
SET_PIECE_AFTER_S = 1.0   # long before its tagged start, or this long after it


def set_piece_catch(dead):
    """(caught, total) graded Smart Tagging set pieces with dead ball near their tagged start.
    An independent check on the dead-ball detector: it uses no ground truth."""
    from . import load
    fps = config.FPS
    graded = load.graded_mask(len(dead))
    starts = [e["start_frame"] for e in load.load_tagging()
              if "set piece" in (e["code"] or "").lower() and 0 <= e["start_frame"] < len(dead)
              and graded[e["start_frame"]]]
    caught = sum(bool(dead[max(0, s - int(SET_PIECE_BEFORE_S * fps)):
                           s + int(SET_PIECE_AFTER_S * fps)].any()) for s in starts)
    return caught, len(starts)


def dead_check(dead, cues):
    """Label-free checks of a dead-ball mask on the provided game, aggregates only. Compared in
    the report with IDSSE's DFL truth (other matches), never with any label of this game."""
    from . import deadball, load
    fps = float(config.FPS)
    graded = load.graded_mask(len(dead))
    halves = [(first, last + 1) for _, first, last in config.half_blocks()]
    stops = []
    for a, b in halves:
        s, e = deadball.runs(dead[a:b])
        stops += [(a + x, a + y) for x, y in zip(s, e) if y - x >= fps]
    lengths = np.array([b - a for a, b in stops]) / fps
    minutes = graded.sum() / fps / 60
    out = np.nan_to_num(cues["beyond"], nan=-1.0) > 0
    seen = cues["ball_seen"]
    blocks = []
    step = int(BLOCK_MIN * 60 * fps)
    for a, b in halves:
        blocks += [float(dead[lo:min(b, lo + step)].mean()) for lo in range(a, b, step)
                   if min(b, lo + step) - lo >= 60 * fps]
    caught, total = set_piece_catch(dead)
    return {
        "dead_share": float(dead[graded].mean()),
        "halves": [float(dead[a:b].mean()) for a, b in halves],
        "blocks": blocks,
        "stops_per90": len(stops) * 90 / minutes,
        "stop_pct": _pct(lengths),
        "under_5s": float((lengths < 5).mean()) if len(lengths) else 0.0,
        "over_60s": float((lengths > 60).mean()) if len(lengths) else 0.0,
        "with_ball_out": float(np.mean([out[a:b].any() for a, b in stops])) if stops else 0.0,
        "dead_if_missing": float(dead[graded & ~seen].mean()),
        "dead_if_seen": float(dead[graded & seen].mean()),
        "missing_if_dead": float((~seen)[graded & dead].mean()),
        "set_pieces": (caught, total),
    }


def format_dead_check(name, r, reference=None):
    pct = lambda d: " ".join("{:.1f}".format(d[p]) for p in PERCENTILES)  # noqa: E731
    lines = [name,
             "  dead share {:.1%}  (halves {})".format(
                 r["dead_share"], " / ".join("{:.1%}".format(h) for h in r["halves"])),
             "  dead share per 15-min block: " + " ".join("{:.0%}".format(b) for b in r["blocks"]),
             "  stoppages {:.0f} per 90, length s p10/25/50/75/90: {}".format(
                 r["stops_per90"], pct(r["stop_pct"])),
             "  stoppages under 5 s {:.0%}, over 60 s {:.0%}, with the ball seen past a line "
             "{:.0%}".format(r["under_5s"], r["over_60s"], r["with_ball_out"]),
             "  dead where the ball is missing {:.0%}, where it is seen {:.0%}; ball missing "
             "on {:.0%} of dead frames".format(r["dead_if_missing"], r["dead_if_seen"],
                                               r["missing_if_dead"]),
             "  Smart Tagging set pieces with dead ball from {:.0f} s before to {:.0f} s after: "
             "{}/{}".format(SET_PIECE_BEFORE_S, SET_PIECE_AFTER_S, *r["set_pieces"])]
    if reference:
        lines.append("  other matches (IDSSE, DFL truth): " + reference)
    return "\n".join(lines)


def detector_breakdown(stoppages, dead):
    """[(rule, count, dead minutes, median length s, share under 5 s)] for detect()'s list,
    each stoppage counted under the rules that found it (before overlaps merge)."""
    fps = float(config.FPS)
    rows = {}
    for a, b, how_start, how_end in stoppages:
        key = "{} / {}".format(how_start, how_end)
        rows.setdefault(key, []).append((int(dead[a:b].sum()), (b - a) / fps))
    out = []
    for key, items in rows.items():
        lengths = np.array([s for _, s in items])
        out.append((key, len(items), sum(f for f, _ in items) / fps / 60,
                    float(np.median(lengths)), float((lengths < 5).mean())))
    return sorted(out, key=lambda r: -r[1])


def format_census(name, c):
    pct = lambda d: " ".join("{:.1f}".format(d[p]) for p in PERCENTILES)  # noqa: E731
    lines = [
        "== {}".format(name),
        "time        match {:.1f} min   dead {:.1f}   live {:.1f}   dead share {:.1%}".format(
            c["match_min"], c["dead_min"], c["live_min"], c["dead_share"]),
        "            by half {}".format("  ".join("{:.1%}".format(s) for s in c["dead_share_halves"])),
    ]
    if c["dead_share_blocks"]:
        lines.append("            by {}' block  {}".format(BLOCK_MIN, "  ".join(
            "{} {:.0%}".format(label, s) for label, s in c["dead_share_blocks"])))
    lines += [
        "possessions {}  ({:.2f}/min)   length s p10/25/50/75/90: {}   <1 s {:.0%}  <2 s {:.0%}".format(
            c["n_poss"], c["poss_per_min"], pct(c["poss_pct"]), c["poss_under_1s"],
            c["poss_under_2s"]),
        "stoppages   {}  ({:.2f}/min)   length s p10/25/50/75/90: {}   <5 s {:.0%}  >60 s {:.0%}".format(
            c["n_stops"], c["stops_per_min"], pct(c["stop_pct"]), c["stops_under_5s"],
            c["stops_over_60s"]),
        "flicker     {} quick returns (<{:.0f} s middle possession), {:.1f} per 100 possessions".format(
            c["flicker"], FLICKER_S, c["flicker_per_100_poss"]),
        "restarts    team changes across a stoppage {} / same team {}".format(
            c["restart_changed"], c["restart_same"]),
        "live spells length s p10/25/50/75/90: {}   possessions per spell: {}".format(
            pct(c["live_pct"]), pct(c["poss_per_live_pct"])),
    ]
    return "\n".join(lines)
