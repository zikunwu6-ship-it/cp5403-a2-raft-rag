"""Evaluate the fine-tuned model across conditions, ablations sweeps.

Each job is skipped if its output JSON already exists, so the script can be
re-run after an interruption without repeating finished work.
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = str(ROOT / ".venv" / "Scripts" / "python.exe")
EVAL = str(ROOT / "src" / "evaluate.py")
RES = ROOT / "results"

MAIN = ROOT / "outputs" / "trainer" / "raft_mixed" / "final"

JOBS = [
    # (out name, adapter, context, k, profile, extra)
    ("raft_closed", MAIN, "closed", 5, "intermediate", []),
    ("raft_rag", MAIN, "retrieved", 5, "intermediate", ["--dump"]),
    ("raft_oracle", MAIN, "oracle", 5, "intermediate", []),
    # retrieval depth
    ("topk_1", MAIN, "retrieved", 1, "intermediate", []),
    ("topk_3", MAIN, "retrieved", 3, "intermediate", []),
    ("topk_8", MAIN, "retrieved", 8, "intermediate", []),
    # learner profile
    ("profile_novice", MAIN, "retrieved", 5, "novice", []),
    ("profile_advanced", MAIN, "retrieved", 5, "advanced", []),
    # ablations
    ("sft_gold", ROOT / "outputs" / "trainer" / "sft_gold" / "final",
     "retrieved", 5, "intermediate", ["--dump"]),
    ("raft_nocot", ROOT / "outputs" / "trainer" / "raft_nocot" / "final",
     "retrieved", 5, "intermediate", ["--dump"]),
    ("raft_random", ROOT / "outputs" / "trainer" / "raft_random" / "final",
     "retrieved", 5, "intermediate", ["--dump"]),
]


def main():
    filters = sys.argv[1:]
    for name, adapter, ctx, k, prof, extra in JOBS:
        if filters and not any(f in name for f in filters):
            continue
        out = RES / f"{name}.json"
        if out.exists():
            print(f"SKIP {name}", flush=True)
            continue
        if not Path(adapter).exists():
            print(f"SKIP {name} (adapter missing: {adapter})", flush=True)
            continue
        cmd = [PY, EVAL, "--condition", name, "--context", ctx,
               "--k", str(k), "--profile", prof,
               "--adapter", str(adapter), "--batch-size", "12",
               "--out", str(out)] + extra
        print("RUN", name, flush=True)
        r = subprocess.run(cmd, cwd=ROOT,
                           stdout=open(ROOT / "outputs" / f"eval_{name}.log", "w"),
                           stderr=subprocess.STDOUT)
        print("DONE", name, r.returncode, flush=True)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
