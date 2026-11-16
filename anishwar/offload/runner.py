"""Runs one stage of the IDSSE searches from the bundle (build.py): stage "a" the nested dead-ball
search (calibrate.run), stage "bc" the possession rules (poss_calibrate.run) and then the change
detector with rule 6 (poss_calibrate.changes_shape_run). Writes aggregates only."""
import glob, json, os, platform, shutil, sys, time
src, work, stage = sys.argv[1], sys.argv[2], sys.argv[3]
pkg = os.path.join(work, "lib")
os.makedirs(pkg, exist_ok=True)
for f in glob.glob(os.path.join(src, "*.py")):
    if os.path.basename(f) != "run.py":
        shutil.copy(f, pkg)
open(os.path.join(pkg, "__init__.py"), "w").close()
os.environ["ARRAYS_DIR"] = src
os.environ["N_WORKERS"] = "4"
sys.path.insert(0, work)
import numpy as np
import pandas as pd
from lib import calibrate as C, deadball, possession, poss_calibrate as P

C.SMOOTH = True
C.WORKERS = 4
cfg = json.load(open(os.path.join(src, "fixed.json")))
deadball.PARAMS.update(cfg["params"])
pr = cfg["profile"]
C.PROFILE = {"bands": np.array(pr["bands"])}
for s in ("dead", "live"):
    C.PROFILE[s] = {"share": pr[s]["share"], "gaps_s": np.array(pr[s]["gaps_s"]), "rate": np.array(pr[s]["rate"])}


def js(x):
    if isinstance(x, dict):
        return {str(k): js(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [js(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    return x


out = {"versions": [platform.python_version(), np.__version__, pd.__version__], "input": cfg["params"],
       "cpus": os.cpu_count()}
path = os.path.join(work, "out_%s.json" % stage)
t = time.time()
if stage == "a":
    r = C.run()
    r.pop("profile")
    out["a"] = r
    out["a_fixed"] = r["best"] == {k: float(v) for k, v in cfg["params"].items()}
    out["a_s"] = time.time() - t
    json.dump(js(out), open(path, "w"))
    print("a done %.0f s, fixed point reproduced: %s" % (out["a_s"], out["a_fixed"]), flush=True)
else:
    r = P.run()
    out["b"] = r
    out["b_s"] = time.time() - t
    json.dump(js(out), open(path, "w"))
    print("b done %.0f s" % out["b_s"], flush=True)
    possession.PARAMS.update(r["best"])
    out["c"] = P.changes_shape_run()
    out["c_s"] = time.time() - t
    json.dump(js(out), open(path, "w"))
    print("c done %.0f s" % out["c_s"], flush=True)
