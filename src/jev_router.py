import json
import os
from datasets import load_dataset
from typesafe_sdk import Choice, TypeSafeClient
from harness import RESULTS

def jev_pick_model(problem_prompt):
    with TypeSafeClient() as client:
        response = client.system_one(
            state={"problem": problem_prompt},
            questions={
                "difficulty": Choice(
                    instructions="Estimate how hard this coding problem is to solve correctly.",
                    criteria={
                        "easy": "Simple, standard logic; a small model should solve this reliably.",
                        "hard": "Requires careful reasoning, edge cases, or non-obvious algorithms; benefits from a stronger model.",
                    },
                ),
            },
        )
    answer = response.answers["difficulty"]
    return answer.choice, answer.probabilities, answer.confidence

def run_request_level_router_from_cache(ds, small_results, large_results, out_path):
    small_by_id = {r["task_id"]: r for r in small_results}
    large_by_id = {r["task_id"]: r for r in large_results}

    results = []
    if os.path.exists(out_path):
        results = json.load(open(out_path))
        done_ids = {r["task_id"] for r in results}
    else:
        done_ids = set()

    for i, problem in enumerate(ds):
        if problem["task_id"] in done_ids:
            continue
        choice, probs, confidence = jev_pick_model(problem["prompt"])
        chosen = large_by_id[problem["task_id"]] if choice == "hard" else small_by_id[problem["task_id"]]
        results.append({
            "task_id": problem["task_id"],
            "jev_choice": choice,
            "jev_probs": probs,
            "jev_confidence": confidence,
            "passed": chosen["passed"],
        })
        print(f"[{i+1}/{len(ds)}] {problem['task_id']}: routed={choice} (conf={confidence:.2f}) → {'PASS' if chosen['passed'] else 'FAIL'}")
        json.dump(results, open(out_path, "w"), indent=2)
    return results

if __name__ == "__main__":
    ds = load_dataset("openai/openai_humaneval")["test"]
    small_results = json.load(open(RESULTS / "results_small.json"))
    large_results = json.load(open(RESULTS / "results_large.json"))

    router_results = run_request_level_router_from_cache(ds, small_results, large_results, RESULTS / "results_router.json")
    router_pass = sum(r["passed"] for r in router_results) / len(router_results)
    hard_count = sum(1 for r in router_results if r["jev_choice"] == "hard")
    print(f"\nRequest-level router pass@1: {router_pass:.3f}")
    print(f"Routed to large: {hard_count}/{len(router_results)} ({hard_count/len(router_results)*100:.0f}%)")