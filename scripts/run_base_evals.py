"""Run the base-model (untuned) evaluation conditions."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = str(ROOT / ".venv" / "Scripts" / "python.exe")

JOBS = [
    ("base_closed", "closed", None),
    ("base_rag", "retrieved", None),
    ("base_oracle", "oracle", None),
]

for cond, ctx, adapter in JOBS:
    cmd = [PY, str(ROOT / "src" / "evaluate.py"),
           "--condition", cond, "--context", ctx, "--batch-size", "12",
           "--out", str(ROOT / "results" / f"{cond}.json"), "--dump"]
    print("RUN", cond, flush=True)
    r = subprocess.run(cmd, cwd=ROOT)
    print("DONE", cond, r.returncode, flush=True)
    if r.returncode != 0:
        sys.exit(r.returncode)
