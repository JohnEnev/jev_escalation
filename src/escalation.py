import json
import os
from datasets import load_dataset
from typesafe_sdk import Noul, TypeSafeClient
from harness import RESULTS

SMALL_COST = 1.0
LARGE_COST = 8.0  # proxy: roughly the 32B / 4B parameter ratio

JUDGE_INSTRUCTIONS = (
    "Does this code correctly and completely solve the stated problem? "
    "Answer no if it is incomplete, a placeholder, or likely has a bug."
)


def score_small_completions(ds, small_by_id, out_path):
    """One Jev call per problem: p(small completion is correct). Cached and resumable."""
    cached = {}
    if os.path.exists(out_path):
        cached = {r["task_id"]: r for r in json.load(open(out_path))}

    with TypeSafeClient() as client:
        for i, problem in enumerate(ds):
            tid = problem["task_id"]
            if tid in cached:
                continue
            small = small_by_id[tid]
            response = client.system_one(
                state={
                    "problem": problem["prompt"],
                    "attempted_solution": small["completion"],
                },
                questions={"looks_correct": Noul(instructions=JUDGE_INSTRUCTIONS)},
            )
            p = response.answers["looks_correct"].noul
            cached[tid] = {
                "task_id": tid,
                "jev_prob_correct": p,
                "small_passed": small["passed"],
            }
            print(f"[{i+1}/{len(ds)}] {tid}: p_correct={p:.2f}  small={'PASS' if small['passed'] else 'FAIL'}")
            json.dump(list(cached.values()), open(out_path, "w"), indent=2)

    return cached


def auroc(scores, labels):
    """Probability that a random positive gets a higher score than a random negative."""
    pos = [s for s, l in zip(scores, labels) if l]
    neg = [s for s, l in zip(scores, labels) if not l]
    if not pos or not neg:
        return float("nan")
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def sweep(task_ids, scores, small_by_id, large_by_id, thresholds):
    rows = []
    n = len(task_ids)
    for t in thresholds:
        passed = 0
        escalated = 0
        for tid in task_ids:
            if scores[tid]["jev_prob_correct"] < t:
                escalated += 1
                passed += large_by_id[tid]["passed"]
            else:
                passed += small_by_id[tid]["passed"]
        rows.append({
            "threshold": t,
            "pass_rate": passed / n,
            "escalation_rate": escalated / n,
            "cost": SMALL_COST + LARGE_COST * escalated / n,
        })
    return rows


if __name__ == "__main__":
    ds = load_dataset("openai/openai_humaneval")["test"]
    small_by_id = {r["task_id"]: r for r in json.load(open(RESULTS / "results_small.json"))}
    large_by_id = {r["task_id"]: r for r in json.load(open(RESULTS / "results_large.json"))}
    task_ids = [p["task_id"] for p in ds]
    n = len(task_ids)

    # 1. Score every small completion once
    scores = score_small_completions(ds, small_by_id, RESULTS / "results_escalation_scores.json")

    # 2. Does Jev's score predict small-model failure?
    fail_scores = [1 - scores[t]["jev_prob_correct"] for t in task_ids]
    fail_labels = [not small_by_id[t]["passed"] for t in task_ids]
    print(f"\nAUROC (Jev predicting small-model failure): {auroc(fail_scores, fail_labels):.3f}")

    # 3. Baselines
    small_pass = sum(small_by_id[t]["passed"] for t in task_ids) / n
    large_pass = sum(large_by_id[t]["passed"] for t in task_ids) / n
    oracle_pass = sum(small_by_id[t]["passed"] or large_by_id[t]["passed"] for t in task_ids) / n

    print("\n=== Baselines ===")
    print(f"{'strategy':<24}{'pass@1':>8}{'cost':>8}")
    print(f"{'always-small':<24}{small_pass:>8.3f}{SMALL_COST:>8.2f}")
    print(f"{'always-large':<24}{large_pass:>8.3f}{LARGE_COST:>8.2f}")
    if os.path.exists(RESULTS / "results_router.json"):
        router = json.load(open(RESULTS / "results_router.json"))
        r_pass = sum(r["passed"] for r in router) / len(router)
        r_hard = sum(r["jev_choice"] == "hard" for r in router) / len(router)
        r_cost = SMALL_COST * (1 - r_hard) + LARGE_COST * r_hard
        print(f"{'request-level router':<24}{r_pass:>8.3f}{r_cost:>8.2f}")
    print(f"{'oracle (ceiling)':<24}{oracle_pass:>8.3f}{'-':>8}")

    # 4. Threshold sweep, no extra API calls
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    rows = sweep(task_ids, scores, small_by_id, large_by_id, thresholds)

    print("\n=== Verify-then-escalate sweep ===")
    print(f"{'threshold':>10}{'pass@1':>10}{'escalated':>12}{'cost':>8}")
    for r in rows:
        print(f"{r['threshold']:>10.1f}{r['pass_rate']:>10.3f}{r['escalation_rate']*100:>11.0f}%{r['cost']:>8.2f}")

    # Sweep is derived from the six published JSONs; print it without adding a seventh file.