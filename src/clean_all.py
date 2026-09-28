"""Trim cached generations and retest HumanEval and HumanEval+ (executes model code).

Run only in a sandbox. --rescore makes 164 new Jev API calls; otherwise existing
Jev scores are retained, and router outcomes are refreshed for each benchmark.
"""
import argparse
import json
import sys
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from datasets import load_dataset
from typesafe_sdk import Noul, TypeSafeClient
from harness import RESULTS, trim_completion

ROOT = RESULTS.parent
FILES = ["results_small.json", "results_large.json",
         "results_continuation_first_line.json", "results_continuation_half.json"]
JUDGE = ("Does this code correctly and completely solve the stated problem? "
         "Answer no if it is incomplete, a placeholder, or likely has a bug.")


def test(problem, completion, timeout):
    code = (problem["prompt"] + completion + "\n\n" + problem["test"] +
            "\n" + f"check({problem['entry_point']})\n")
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code)
        path = Path(f.name)
    try:
        r = subprocess.run([sys.executable, str(path)], capture_output=True,
                           text=True, timeout=timeout)
        return r.returncode == 0, (r.stderr[-500:] if r.returncode else None)
    except subprocess.TimeoutExpired:
        return False, "timeout"
    finally:
        path.unlink()


def main(rescore=False):
    base = {p["task_id"]: p for p in load_dataset("openai/openai_humaneval")["test"]}
    plus = {p["task_id"]: p for p in load_dataset("evalplus/humanevalplus")["test"]}
    cleaned = {}
    for filename in FILES:
        # raw_completion in published results preserves the original generation.
        rows = json.loads((RESULTS / filename).read_text())
        for bench, problems, folder, timeout in [
            ("humaneval", base, RESULTS, 10),
            ("humanevalplus", plus, ROOT / "humanevalplus", 30),
        ]:
            folder.mkdir(parents=True, exist_ok=True)
            def score(row):
                raw = row.get("raw_completion", row["completion"])
                body = trim_completion(raw)
                passed, error = test(problems[row["task_id"]], body, timeout)
                return {**row, "raw_completion": raw, "completion": body,
                        "passed": passed, "error": error}
            with ThreadPoolExecutor(max_workers=8) as pool:
                new = list(pool.map(score, rows))
            (folder / filename).write_text(json.dumps(new, indent=2))
            cleaned[(bench, filename)] = {r["task_id"]: r for r in new}
            print(f"{bench}: {filename}: {sum(r['passed'] for r in new)}/{len(new)}")

    if rescore:
        scores = []
        with TypeSafeClient() as client:
            for tid, row in cleaned[("humaneval", "results_small.json")].items():
                resp = client.system_one(
                    state={"problem": base[tid]["prompt"], "attempted_solution": row["completion"]},
                    questions={"looks_correct": Noul(instructions=JUDGE)},
                )
                scores.append({"task_id": tid, "jev_prob_correct": resp.answers["looks_correct"].noul})
        for folder in [RESULTS, ROOT / "humanevalplus"]:
            (folder / "results_escalation_scores.json").write_text(json.dumps(scores, indent=2))

    router = json.loads((RESULTS / "results_router.json").read_text())
    for bench, folder in [("humaneval", RESULTS), ("humanevalplus", ROOT / "humanevalplus")]:
        s = cleaned[(bench, "results_small.json")]
        l = cleaned[(bench, "results_large.json")]
        rows = [{**r, "passed": (l if r["jev_choice"] == "hard" else s)[r["task_id"]]["passed"]}
                for r in router]
        (folder / "results_router.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rescore", action="store_true", help="repeat Jev scoring (API calls)")
    main(parser.parse_args().rescore)
