"""Score a submission and say *why* it scores what it does.

The headline numbers come from the repository's own evaluation/possession_metrics.evaluate, so
they are the official numbers by construction. On top of them this adds an error taxonomy. On
the provided game it is a correctness check only (RULES.md: the public window is not a tuning
set); on other matches with labels it is a full evaluation:

    stoppages     each GT dead-ball stretch: found or missed, and how far off its edges are;
                  each predicted one with no GT counterpart: spurious
    changes       each GT possession change (H<->A): found within a tolerance or missed, and the
                  latency of the ones found; each predicted change with no GT counterpart: spurious
    possessions   each GT possession, by what happened to it: hit, shifted, split, merged, missed

Everything temporal runs on the abstention-filled prediction inside the graded runs, exactly as
the scorer does. All printed output is aggregate; the per-event list goes to a local file only.
"""

import os
import sys

import numpy as np

from . import config

sys.path.insert(0, config.EVALUATION_DIR)
import possession_metrics as pm  # noqa: E402

H, A, D, X = (pm.LABELS.index(label) for label in "HADX")
CHANGE_TOL = 50          # frames: a change found within 2 s counts as found, latency reported
STOP_MIN_OVERLAP = 0.5   # share of a GT stoppage a predicted one must cover to count as found
HIT_IOU = 0.7            # the strictest spellF1 threshold
MISS_IOU = 0.3           # the loosest
LATENCY_BINS = (1, 3, 8, 25, 50)  # frames; 1 is free, 8 costs ~0.1 spellF1, 25 is one second


def load_gt(name="public"):
    path = config.PUBLIC_GT if name == "public" else name
    return pm.label_codes(pm.read_label_csv(path))


def load_pred(path):
    frame = pm.read_label_csv(path)
    problem = pm.validate_submission(frame, config.N_FRAMES)
    if problem:
        raise ValueError("{}: {}".format(os.path.basename(path), problem))
    return pm.label_codes(frame)


def _overlap(a, b):
    return max(0, min(a[1], b[1]) - max(a[0], b[0]))


def stoppage_report(gt_segs, pred_segs):
    gt_stops = [s for s in gt_segs if s[2] == D]
    pred_stops = [s for s in pred_segs if s[2] == D]
    found, missed, start_off, end_off, events = 0, 0, [], [], []
    used = set()
    for g in gt_stops:
        best = max(((_overlap(g, p), j) for j, p in enumerate(pred_stops)), default=(0, None))
        if best[1] is not None and best[0] >= STOP_MIN_OVERLAP * (g[1] - g[0]):
            p = pred_stops[best[1]]
            used.add(best[1])
            found += 1
            start_off.append(p[0] - g[0])
            end_off.append(p[1] - g[1])
            events.append(("stop_found", g[0], p[0] - g[0], p[1] - g[1]))
        else:
            missed += 1
            events.append(("stop_missed", g[0], None, None))
    unused = [p for j, p in enumerate(pred_stops) if j not in used]
    spurious = [p for p in unused if not any(_overlap(g, p) for g in gt_stops)]
    partial = [p for p in unused if any(_overlap(g, p) for g in gt_stops)]
    events += [("stop_spurious", p[0], None, None) for p in spurious]
    events += [("stop_partial", p[0], None, None) for p in partial]
    return {"n_gt": len(gt_stops), "n_pred": len(pred_stops), "found": found, "missed": missed,
            "spurious": len(spurious), "partial": len(partial),
            "start_offsets": start_off, "end_offsets": end_off}, events


def change_report(gt_trans, pred_trans):
    def changes(trans):
        return [t for t in trans if {t[1], t[2]} == {H, A}]
    gt_changes, pred_changes = changes(gt_trans), changes(pred_trans)
    pairs, missed, spurious = pm.match_transitions(gt_changes, pred_changes, CHANGE_TOL)
    latencies = [p[0] - g[0] for g, p in pairs]
    events = [("change_found", g[0], p[0] - g[0], None) for g, p in pairs]
    events += [("change_missed", g[0], None, None) for g in missed]
    events += [("change_spurious", p[0], None, None) for p in spurious]
    return {"n_gt": len(gt_changes), "n_pred": len(pred_changes), "found": len(pairs),
            "missed": len(missed), "spurious": len(spurious), "latencies": latencies}, events


def possession_report(gt_spells, pred_spells):
    """Fate of each GT possession, and count of predicted ones matching nothing."""
    fates = {"hit": 0, "shifted": 0, "split": 0, "merged": 0, "missed": 0}
    events = []
    for g in gt_spells:
        same = [p for p in pred_spells if p[2] == g[2] and _overlap(g, p)]
        ious = [_overlap(g, p) / float(max(g[1], p[1]) - min(g[0], p[0])) for p in same]
        best = max(ious, default=0.0)
        big_pieces = [p for p in same if _overlap(g, p) >= 0.1 * (g[1] - g[0])]
        if best >= HIT_IOU:
            fate = "hit"
        elif len(big_pieces) >= 2:
            fate = "split"
        elif same and best < MISS_IOU and any(
                sum(1 for q in gt_spells if q[2] == p[2] and _overlap(q, p)) >= 2 for p in same):
            fate = "merged"
        elif best >= MISS_IOU:
            fate = "shifted"
        else:
            fate = "missed"
        fates[fate] += 1
        events.append(("possession_" + fate, g[0], round(best, 3), None))
    spurious = sum(1 for p in pred_spells
                   if not any(g[2] == p[2] and _overlap(g, p) for g in gt_spells))
    return {"fates": fates, "spurious": spurious}, events


def score(pred_codes, gt_codes, bootstrap_n=0):
    """Official metrics plus the taxonomy, and the per-event list for local inspection."""
    official = pm.evaluate(gt_codes, pred_codes, bootstrap_n=bootstrap_n)
    runs = pm.valid_runs(gt_codes)
    filled = pm.fill_abstentions(pred_codes, runs)
    graded = np.zeros(len(gt_codes), bool)
    for start, stop in runs:
        graded[start:stop] = True
    pred_graded = np.where(graded, filled, X)

    gt_segs, pred_segs = pm.segments(gt_codes, runs), pm.segments(pred_graded, runs)
    stops, e1 = stoppage_report(gt_segs, pred_segs)
    changes, e2 = change_report(pm.transitions(gt_codes, runs), pm.transitions(pred_graded, runs))
    possessions, e3 = possession_report(pm.possession_spells(gt_codes, runs),
                                        pm.possession_spells(pred_graded, runs))
    events = sorted(e1 + e2 + e3, key=lambda e: e[1])
    return {"official": official, "stops": stops, "changes": changes,
            "possessions": possessions,
            "confusion": pm.scored_confusion(pm.confusion(gt_codes, pred_codes))}, events


def _histogram(values):
    if not values:
        return "none"
    a = np.abs(np.asarray(values))
    parts, low = [], -1
    for high in LATENCY_BINS:
        parts.append("<={}:{}".format(high, int(((a > low) & (a <= high)).sum())))
        low = high
    parts.append(">{}:{}".format(LATENCY_BINS[-1], int((a > LATENCY_BINS[-1]).sum())))
    return "  ".join(parts)


def _signed(values):
    if not values:
        return "n/a"
    v = np.asarray(values)
    return "median {:+.0f}  mean {:+.1f}  early {}  late {}  exact {}".format(
        np.median(v), v.mean(), int((v < 0).sum()), int((v > 0).sum()), int((v == 0).sum()))


def format_report(name, result):
    o, s, c, p = result["official"], result["stops"], result["changes"], result["possessions"]
    cm = result["confusion"]
    lines = [
        "== {}".format(name),
        "spellF1 {:.4f}   macroF1 {:.4f}   (official)".format(o["spell_f1_mean"], o["macro_f1"]),
    ]
    if o.get("spell_bootstrap_low") is not None:
        lines.append("spellF1 bootstrap 5-95%  {:.4f} - {:.4f}".format(
            o["spell_bootstrap_low"], o["spell_bootstrap_high"]))
    lines += [
        "possessions  gt {} (mean {:.1f} s)   pred {} (mean {:.1f} s)".format(
            o["n_gt_spells"], o["gt_mean_spell_duration_s"],
            o["n_pred_spells"], o["mean_spell_duration_s"]),
        "confusion (rows gt H/A/D, cols pred H/A/D/X)",
    ]
    lines += ["  {}  {}".format(label, "  ".join("{:>6}".format(int(n)) for n in row))
              for label, row in zip("HAD", cm)]
    lines += [
        "stoppages    gt {}  pred {}  found {}  missed {}  spurious {}  partial {}".format(
            s["n_gt"], s["n_pred"], s["found"], s["missed"], s["spurious"], s["partial"]),
        "  start offset (frames)  " + _signed(s["start_offsets"]),
        "  end offset   (frames)  " + _signed(s["end_offsets"]),
        "changes H<->A  gt {}  pred {}  found {}  missed {}  spurious {}  (within {} frames)".format(
            c["n_gt"], c["n_pred"], c["found"], c["missed"], c["spurious"], CHANGE_TOL),
        "  latency (frames)       " + _signed(c["latencies"]),
        "  |latency| histogram    " + _histogram(c["latencies"]),
        "gt possessions by fate   " + "  ".join(
            "{} {}".format(k, v) for k, v in p["fates"].items()),
        "pred possessions matching no gt possession  {}".format(p["spurious"]),
    ]
    return "\n".join(lines)


def write_events(path, events):
    """Per-event list for watching in the video. Local and gitignored; never printed."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as handle:
        handle.write("kind,frame,video_time,value,value2\n")
        for kind, frame, value, value2 in events:
            seconds = frame / float(config.FPS)
            handle.write("{},{},{}:{:05.2f},{},{}\n".format(
                kind, frame, int(seconds // 60), seconds % 60,
                "" if value is None else value, "" if value2 is None else value2))
