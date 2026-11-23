"""Everything that has to happen after the main adapter is trained.

Sequenced for wall-clock, not for tidiness: the headline conditions are
evaluated first so that the report's main numbers exist within minutes of the
adapter appearing, and the two-hour ablation training runs afterwards while
those results are being read.
"""
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = str(ROOT / ".venv" / "Scripts" / "python.exe")

STEPS = [
    ("headline evals (C3, C4, C5)", [PY, "scripts/run_evals.py",
                                     "raft_closed", "raft_rag", "raft_oracle"]),
    ("ablation training (A1, A2, A3)", [PY, "scripts/run_training.py"]),
    ("remaining evals", [PY, "scripts/run_evals.py"]),
    ("artefacts", [PY, "scripts/finalize.py"]),
]

for label, cmd in STEPS:
    print(f"\n########## {label} ##########", flush=True)
    t0 = time.time()
    r = subprocess.run(cmd, cwd=ROOT)
    print(f"########## {label} rc={r.returncode} "
          f"({(time.time()-t0)/60:.1f} min) ##########", flush=True)
    if r.returncode != 0:
        sys.exit(r.returncode)

print("\nREST COMPLETE", flush=True)
