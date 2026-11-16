"""Builds the bundle the offloaded searches run from (a private Kaggle dataset).

    cd anishwar && .venv/bin/python offload/build.py OUT_DIR FIXED_JSON

Only what the IDSSE searches need: the calibration modules with docstrings, comments and every
function the searches do not call removed, neutral stubs for the modules that read our match,
the organisers' metric code, the seven smoothed IDSSE cue caches renamed a0..a6, and FIXED_JSON
(offload/fixed_point.py: the dead-ball constants and the hiding profile, aggregates only). Nothing
from our match's tracking, tagging or labels goes in.
"""
import ast, os, re, shutil, sys
sys.path.insert(0, ".")
from mtah import external  # noqa: E402
OUT, FIXED = sys.argv[1], sys.argv[2]
MODS = ("calibrate", "deadball", "shape", "challenger", "possession", "poss_calibrate", "changes")
MIDS = external.MATCHES


def strip(src):
    t = ast.parse(src)
    for node in ast.walk(t):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.ClassDef)) and node.body \
                and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant) \
                and isinstance(node.body[0].value.value, str):
            node.body = node.body[1:] or [ast.Pass()]
    return ast.unparse(t) + "\n"


DROP = {"calibrate": {"our_cues", "_fmt", "report"},
        "challenger": {"run", "report"},
        "poss_calibrate": {"report", "shape_inputs", "shape_fit", "shape_run", "shape_report",
                           "owner_run", "owner_report", "changes_run", "changes_report",
                           "changes_shape_report"}}
RENAME = {"idsse_cues": "cues_of", "idsse_press": "press_of", "_idsse_tracks": "_tracks",
          "SMOOTH_IDSSE": "SMOOTH", "MTAH_WORKERS": "N_WORKERS"}


def prune(src, drop):
    t = ast.parse(src)
    t.body = [n for n in t.body if not (isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in drop)]
    return ast.unparse(t) + "\n"


def put(name, src):
    open(os.path.join(OUT, name), "w").write(src)


for m in MODS:
    s = prune(strip(open("mtah/%s.py" % m).read()), DROP.get(m, ()))
    for a, b in RENAME.items():
        s = re.sub(r"\b%s\b" % a, b, s)
    s = s.replace("deadball.ball_gap_profile(*our_cues())", "None")
    s = s.replace("from . import config, deadball, external, load, shape",
                  "from . import deadball, external, shape")
    put(m + ".py", s)
put("metric.py", strip(open("../evaluation/possession_metrics.py").read()))
put("external.py", "import os\nMATCHES = %r\nARRAYS_DIR = os.environ.get('ARRAYS_DIR', '.')\n"
    % (tuple("a%d" % i for i in range(7)),))
put("harness.py", "from . import metric as pm\nH, A, D, X = (pm.LABELS.index(l) for l in 'HADX')\n")
for i, mid in enumerate(MIDS):
    for kind in ("cues5s9-3", "press1s9-3"):
        shutil.copy("external/arrays/%s_%s.npz" % (mid, kind), os.path.join(OUT, "a%d_%s.npz" % (i, kind)))
shutil.copy(FIXED, os.path.join(OUT, "fixed.json"))
