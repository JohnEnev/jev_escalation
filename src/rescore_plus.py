# rescore_plus.py
import json
import os
import shutil
import subprocess
import tempfile
import sys
from concurrent.futures import ThreadPoolExecutor
from datasets import load_dataset
from harness import RESULTS

TIMEOUT = 30  # HumanEval+ tests are much larger; 6s would cause false timeouts
OUT_DIR = RESULTS.parent / "humanevalplus"

SOURCES = {
    "results_small.json": "results_small.json",
    "results_large.json": "results_large.json",
    "results_continuation_first_line.json": "results_continuation_first_line.json",
    "results_continuation_half.json": "results_continuation_half.json",
}

def run_tests(problem, completion):
    code = problem["prompt"] + completion + "\n\n" + problem["test"] + "\n" + f"check({problem['entry_point']})\n"
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code)
        path = f.name
    try:
        r = subprocess.run([sys.executable, path], capture_output=True, text=True, timeout=TIMEOUT)
        return r.returncode == 0, (r.stderr[-500:] if r.returncode != 0 else None)
    except subprocess.TimeoutExpired:
        return False, "timeout"
    finally:
        os.unlink(path)

plus = {p["task_id"]: p for p in load_dataset("evalplus/humanevalplus")["test"]}
os.makedirs(OUT_DIR, exist_ok=True)
rescored = {}

for src, dst in SOURCES.items():
    rows = json.load(open(RESULTS / src))

    def score(row):
        passed, error = run_tests(plus[row["task_id"]], row["completion"])
        return {**row, "passed_base": row["passed"], "passed": passed, "error": error}

    with ThreadPoolExecutor(max_workers=8) as pool:
        new_rows = list(pool.map(score, rows))

    json.dump(new_rows, open(os.path.join(OUT_DIR, dst), "w"), indent=2)
    rescored[dst] = {r["task_id"]: r for r in new_rows}

    base = sum(r["passed_base"] for r in new_rows) / len(new_rows)
    strict = sum(r["passed"] for r in new_rows) / len(new_rows)
    flipped = sum(r["passed_base"] and not r["passed"] for r in new_rows)
    timeouts = sum(r["error"] == "timeout" for r in new_rows)
    print(f"{dst:<40} base {base:.3f} -> plus {strict:.3f}  ({flipped} flipped to fail, {timeouts} timeouts)")

# Jev scores are unchanged: same answers were judged
shutil.copy(RESULTS / "results_escalation_scores.json", OUT_DIR / "results_escalation_scores.json")

# Router results store 'passed' directly, so recompute from its routing choices
small_p = rescored["results_small.json"]
large_p = rescored["results_large.json"]
router = json.load(open(RESULTS / "results_router.json"))
for r in router:
    src = large_p if r["jev_choice"] == "hard" else small_p
    r["passed"] = src[r["task_id"]]["passed"]
json.dump(router, open(os.path.join(OUT_DIR, "results_router.json"), "w"), indent=2)

print(f"\nWrote rescored files to ./{OUT_DIR}/")