"""Writes the offload notebook, one per stage, with runner.py embedded:

    python offload/make_nb.py OUT_DIR a bc

Each notebook installs uv and then Python 3.12.8, numpy 1.26.3 and pandas 2.1.4 into its own
environment, every wheel pinned by its PyPI sha256, so the run uses the same versions as ours;
then runs one stage and keeps only its aggregate results file (out_<stage>.json)."""
import json, os, sys
RUNNER = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "runner.py")).read()
setup = r'''STAGE = "%s"   # "a": nested dead-ball search; "bc": possession rules, then the change detector
import glob, os, subprocess, sys
src = os.path.dirname(glob.glob("/kaggle/input/**/fixed.json", recursive=True)[0])
uv = [sys.executable, "-m", "uv"]
open("/tmp/uv.txt", "w").write('uv==0.5.11 --hash=sha256:08c660c69e7dd874b52ad96b597b57e9f999659f3d9827cdbad884a68e48f7e9\n')
open("/tmp/req.txt", "w").write('numpy==1.26.3 --hash=sha256:5f24750ef94d56ce6e33e4019a8a4d68cfdb1ef661a52cdaee628a56d2437419\npandas==2.1.4 --hash=sha256:e9f17f2b6fc076b2a0078862547595d66244db0f41bf79fc5f64a5c4d635bead\npython-dateutil==4.0.0 --hash=sha256:a5e246f79eb80475e691bf93862cead98decd8a498cf53ba0516fcdc0f50dc1c\npytz==2026.6 --hash=sha256:7fed3fee8fd6bd4a2b7b75a8d56021d0624127bf69bd5691f115db929ac570b3\ntzdata==2026.6 --hash=sha256:0a10eec6e13ef16b627fabed9a5f2ba61a62b1c67f327305f4e7e412a70b4aec\n')
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--require-hashes", "--only-binary", ":all:", "-r", "/tmp/uv.txt"], check=True)
subprocess.run(uv + ["venv", "--python", "3.12.8", "/tmp/env"], check=True)
subprocess.run(uv + ["pip", "install", "--python", "/tmp/env/bin/python", "--require-hashes", "--no-build", "-r", "/tmp/req.txt"], check=True)
subprocess.run(["/tmp/env/bin/python", "-c", "import sys, numpy, pandas; print(sys.version.split()[0], numpy.__version__, pandas.__version__)"], check=True)'''
run = r'''env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
subprocess.run(["/tmp/env/bin/python", "-u", "-W", "ignore", "/tmp/run.py", src, "/kaggle/working", STAGE],
               check=True, env=env)
for p in glob.glob("/kaggle/working/*"):
    if not os.path.basename(p).startswith("out_"):
        subprocess.run(["rm", "-rf", p])'''
cell = lambda s: {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": s.splitlines(True)}
for stage in sys.argv[2:]:
    nb = {"cells": [cell(setup % stage), cell("%%writefile /tmp/run.py\n" + RUNNER), cell(run)],
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
    d = os.path.join(sys.argv[1], stage); os.makedirs(d, exist_ok=True)
    json.dump(nb, open(os.path.join(d, "offload.ipynb"), "w"), indent=1)
