"""Rebuild every derived artefact from the current result files.

Order matters: tables and figures are produced from results/*.json, then the
documents and the deck are produced from those tables and figures. Running this
after any new evaluation keeps the prose, tables, figures and slides consistent.
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = str(ROOT / ".venv" / "Scripts" / "python.exe")
PYPLOT = r"C:\Users\31558\.workbuddy-ai\binaries\python\envs\default\Scripts\python.exe"

STEPS = [
    ("analyse (tables + retrieval metrics)", [PY, "src/analyse.py"]),
    ("figures", [PYPLOT, "src/make_figures.py"]),
    ("docx reports", [PY, "src/build_docs.py"]),
    ("ai-use template", [PY, "src/fill_ai_template.py"]),
    ("presentation", [PY, "src/build_pptx.py"]),
]

for label, cmd in STEPS:
    print(f"\n=== {label} ===", flush=True)
    r = subprocess.run(cmd, cwd=ROOT)
    if r.returncode != 0:
        print(f"FAILED at: {label}", flush=True)
        sys.exit(r.returncode)
print("\nALL ARTEFACTS REBUILT", flush=True)
