"""The dead-ball fixed point, run locally (the hiding profile is measured on our match):

    cd anishwar && .venv/bin/python offload/fixed_point.py FIXED_JSON

Searches the dead-ball constants on all seven IDSSE matches, re-measures the hiding profile with
the result, and repeats until the search returns its own input. Writes the constants and that
profile (shares, band rates, gap lengths: aggregates only) to FIXED_JSON for the offloaded run."""
import sys, time, json, warnings; sys.path.insert(0, "."); warnings.simplefilter("ignore")
import numpy as np
from mtah import calibrate as C, deadball, external
t = time.time(); every = list(range(7)); seen = []
for rnd in range(1, 6):
    ev = C.Evaluator(external.MATCHES, True)
    cur = tuple(int(abs(C.RANGES[k] - deadball.PARAMS[k]).argmin()) for k in C.KEYS)
    best, bo = C.search(ev, every)
    print("round %d: start %.4f -> found %.4f  changed: %s  (%.0f s)" % (rnd, ev.objective(cur, every), bo,
          {k: (deadball.PARAMS[k], v) for k, v in ev.params(best).items() if v != deadball.PARAMS[k]}, time.time() - t), flush=True)
    if best == cur:
        print("FIXED POINT in round %d" % rnd); break
    if best in seen:
        print("CYCLE"); sys.exit(1)
    seen.append(cur); deadball.PARAMS.update(ev.params(best))
else:
    print("NO FIXED POINT"); sys.exit(1)
prof = ev.profile
js = {"bands": [float(x) for x in prof["bands"]]}
for s in ("dead", "live"):
    js[s] = {"share": prof[s]["share"], "gaps_s": [round(float(x), 2) for x in prof[s]["gaps_s"]],
             "rate": [float(x) for x in prof[s]["rate"]]}
json.dump({"params": dict(deadball.PARAMS), "profile": js}, open(sys.argv[1], "w"))
print("wrote", sys.argv[1], "%.0f s" % (time.time() - t))
