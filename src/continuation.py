import json
import os
import sys
from datasets import load_dataset
from harness import generate, run_tests, LARGE_URL, LARGE_MODEL, RESULTS, trim_completion

STOP = ["\nclass ", "\ndef ", "\nif __name__", "\nprint("]


def make_prefix(completion, variant):
    lines = completion.split("\n")
    while lines and not lines[0].strip():   # drop leading blank lines
        lines.pop(0)
    while lines and not lines[-1].strip():  # drop trailing blank lines
        lines.pop()
    if variant == "first_line":
        k = min(1, len(lines))
    elif variant == "half":
        k = len(lines) // 2   # a one-line completion keeps 0 lines, i.e. cold large
    else:
        raise ValueError(variant)
    kept = lines[:k]
    return ("\n".join(kept) + "\n") if kept else ""


def run_continuation(ds, small_by_id, variant, out_path):
    results = {}
    if os.path.exists(out_path):
        results = {r["task_id"]: r for r in json.load(open(out_path))}

    for i, problem in enumerate(ds):
        tid = problem["task_id"]
        if tid in results:
            continue
        prefix = make_prefix(small_by_id[tid]["completion"], variant)
        cont, finish_reason = generate(LARGE_URL, LARGE_MODEL, problem["prompt"] + prefix, stop=STOP)
        full = trim_completion(prefix + cont)
        passed, error = run_tests(problem, full)
        results[tid] = {
            "task_id": tid,
            "variant": variant,
            "prefix_lines": prefix.count("\n"),
            "completion": full,
            "finish_reason": finish_reason,
            "passed": passed,
            "error": error,
        }
        print(f"[{i+1}/{len(ds)}] {tid} ({variant}, {prefix.count(chr(10))} lines kept): {'PASS' if passed else 'FAIL'}")
        json.dump(list(results.values()), open(out_path, "w"), indent=2)

    return results


if __name__ == "__main__":
    ds = load_dataset("openai/openai_humaneval")["test"]
    small_by_id = {r["task_id"]: r for r in json.load(open(RESULTS / "results_small.json"))}
    variants = sys.argv[1:] or ["first_line", "half"]

    for v in variants:
        print(f"\n=== Continuation: {v} ===")
        res = run_continuation(ds, small_by_id, v, RESULTS / f"results_continuation_{v}.json")
        rate = sum(r["passed"] for r in res.values()) / len(res)
        print(f"Unconditional pass@1 ({v}): {rate:.3f}")