"""Train the three adapters used in the report, sequentially, on one GPU.

Order matters: the main RAFT model is trained first so that evaluation can begin
on it while the ablations are still queued.
"""
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = str(ROOT / ".venv" / "Scripts" / "python.exe")

RUNS = [
    # (run name, data tag, extra args)
    ("raft_mixed", "mixed", []),
    ("sft_gold", "gold_only", []),
    ("raft_nocot", "mixed", ["--answer-only"]),
    ("raft_random", "random", []),
]

for run, tag, extra in RUNS:
    meta = ROOT / "outputs" / "trainer" / run / "meta.json"
    if meta.exists():
        print(f"SKIP {run} (already trained)", flush=True)
        continue
    cmd = [PY, str(ROOT / "src" / "train_qlora.py"),
           "--data-tag", tag, "--run-name", run, "--epochs", "1"] + extra
    print("TRAIN", run, " ".join(cmd), flush=True)
    t0 = time.time()
    log = ROOT / "outputs" / f"train_{run}.log"
    with log.open("w", encoding="utf-8") as fh:
        r = subprocess.run(cmd, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT)
    print(f"DONE {run} rc={r.returncode} in {(time.time()-t0)/60:.1f} min", flush=True)
    if r.returncode != 0:
        print("FAILED - see", log, flush=True)
        sys.exit(r.returncode)

print("ALL DONE", flush=True)
