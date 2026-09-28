import json
import subprocess
import tempfile
import os
import sys
from pathlib import Path
import requests
from datasets import load_dataset

SMALL_URL = "http://localhost:8000/v1/completions"
LARGE_URL = "http://localhost:8002/v1/completions"
SMALL_MODEL = "/workspace/models/qwen3-4b"
LARGE_MODEL = "/workspace/models/qwen3-32b-awq"
RESULTS = Path(__file__).resolve().parents[1] / "results" / "humaneval"


def trim_completion(completion):
    """HumanEval expects only the indented body, not a subsequent top-level definition."""
    lines = []
    for line in completion.split("\n"):
        if line.strip() and not line[0].isspace():
            break
        lines.append(line)
    return "\n".join(lines)

def generate(url, model, prompt, max_tokens=512, temperature=0.0, stop=None):
    resp = requests.post(url, json={
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stop": stop,
    }, timeout=60)
    if resp.status_code != 200:
        print("ERROR BODY:", resp.text)
    resp.raise_for_status()
    choice = resp.json()["choices"][0]
    text = choice["text"]
    if "```" in text:
        text = text.split("```")[0]
    return trim_completion(text), choice["finish_reason"]

def run_tests(problem, completion, timeout=6):
    full_code = (
        problem["prompt"] + completion + "\n\n" +
        problem["test"] + "\n" +
        f"check({problem['entry_point']})\n"
    )
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(full_code)
        path = f.name
    try:
        result = subprocess.run(
            [sys.executable, path],
            capture_output=True, text=True, timeout=timeout
        )
        passed = result.returncode == 0
        error = result.stderr if not passed else None
    except subprocess.TimeoutExpired:
        passed = False
        error = "timeout"
    finally:
        os.unlink(path)
    return passed, error

def run_baseline(ds, url, model, out_path):
    results = []
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    if os.path.exists(out_path):
        results = json.load(open(out_path))
        done_ids = {r["task_id"] for r in results}
    else:
        done_ids = set()

    for i, problem in enumerate(ds):
        if problem["task_id"] in done_ids:
            continue
        completion, finish_reason = generate(
            url, model, problem["prompt"],
            stop=["\nclass ", "\ndef ", "\nif __name__", "\nprint("]
        )
        passed, error = run_tests(problem, completion)
        results.append({
            "task_id": problem["task_id"],
            "completion": completion,
            "finish_reason": finish_reason,
            "passed": passed,
            "error": error,
        })
        print(f"[{i+1}/{len(ds)}] {problem['task_id']}: {'PASS' if passed else 'FAIL'}")
        json.dump(results, open(out_path, "w"), indent=2)
    return results

if __name__ == "__main__":
    ds = load_dataset("openai/openai_humaneval")["test"]

    print("=== Always-small baseline ===")
    small_results = run_baseline(ds, SMALL_URL, SMALL_MODEL, RESULTS / "results_small.json")
    small_pass = sum(r["passed"] for r in small_results) / len(small_results)
    print(f"Small model pass@1: {small_pass:.3f}")

    print("=== Always-large baseline ===")
    large_results = run_baseline(ds, LARGE_URL, LARGE_MODEL, RESULTS / "results_large.json")
    large_pass = sum(r["passed"] for r in large_results) / len(large_results)
    print(f"Large model pass@1: {large_pass:.3f}")