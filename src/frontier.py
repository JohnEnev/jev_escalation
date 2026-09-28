# frontier.py
import json
import random
import argparse
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SMALL_COST, LARGE_COST = 1.0, 8.0
THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
B = 2000
random.seed(0)

parser = argparse.ArgumentParser(description="Paired bootstrap comparison and legacy frontier")
parser.add_argument("--benchmark", choices=["humaneval", "humanevalplus"], default="humaneval")
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
data = root / "results" / args.benchmark
load = lambda name: {r["task_id"]: r for r in json.loads((data / name).read_text())}
small, large = load("results_small.json"), load("results_large.json")
fl, half = load("results_continuation_first_line.json"), load("results_continuation_half.json")
scores = load("results_escalation_scores.json")
router = load("results_router.json")
tids = list(small)
n = len(tids)

def is_stub(c):
    s = c.strip().lower()
    return s in ("pass", "") or "your code here" in s

def outcomes(strategy, t=None):
    """Per-problem list of (passed, cost)."""
    out = []
    for x in tids:
        esc = t is not None and scores[x]["jev_prob_correct"] < t
        if strategy == "always_small":
            out.append((small[x]["passed"], SMALL_COST))
        elif strategy == "always_large":
            out.append((large[x]["passed"], LARGE_COST))
        elif strategy == "router":
            hard = router[x]["jev_choice"] == "hard"
            out.append((router[x]["passed"], LARGE_COST if hard else SMALL_COST))
        elif strategy == "cascade":
            out.append((large[x]["passed"], SMALL_COST + LARGE_COST) if esc else (small[x]["passed"], SMALL_COST))
        elif strategy == "continue_half":
            out.append((half[x]["passed"], SMALL_COST + LARGE_COST) if esc else (small[x]["passed"], SMALL_COST))
        elif strategy == "hybrid":
            if not esc:
                out.append((small[x]["passed"], SMALL_COST))
            elif is_stub(large[x]["completion"]):
                out.append((fl[x]["passed"], SMALL_COST + 2 * LARGE_COST))
            else:
                out.append((large[x]["passed"], SMALL_COST + LARGE_COST))
    return out

BOOT = [[random.randrange(n) for _ in range(n)] for _ in range(B)]

def mean_pass(v, idx=None):
    idx = idx if idx is not None else range(n)
    return sum(v[i][0] for i in idx) / n

def mean_cost(v):
    return sum(c for _, c in v) / n

def ci(v):
    samples = sorted(mean_pass(v, idx) for idx in BOOT)
    return samples[int(0.025 * B)], samples[int(0.975 * B)]

def paired_diff(a, b):
    """Mean and 95% CI of pass(a) - pass(b), resampling the same problems for both."""
    diffs = sorted(mean_pass(a, idx) - mean_pass(b, idx) for idx in BOOT)
    return mean_pass(a) - mean_pass(b), diffs[int(0.025 * B)], diffs[int(0.975 * B)]

# --- Paired comparisons ---
print("Paired differences in pass@1 (95% bootstrap CI):")
for t in [0.5, 0.6]:
    comps = [
        ("hybrid", "cascade"),
        ("hybrid", "always_small"),
        ("hybrid", "always_large"),
        ("cascade", "router"),
        ("cascade", "continue_half"),
    ]
    for a, b in comps:
        va = outcomes(a, t if a in ("cascade", "hybrid", "continue_half") else None)
        vb = outcomes(b, t if b in ("cascade", "hybrid", "continue_half") else None)
        d, lo, hi = paired_diff(va, vb)
        flag = "" if lo <= 0 <= hi else "  *"
        print(f"  t={t}  {a:>14} - {b:<14} {d:+.3f}  [{lo:+.3f}, {hi:+.3f}]{flag}")
print("  (* = interval excludes zero)")

# --- Chart ---
fig, ax = plt.subplots(figsize=(10, 5.5))

for name, label, style in [
    ("cascade", "Verify, then restart (cascade)", "-o"),
    ("hybrid", "Cascade + stub retry (hybrid)", "-s"),
    ("continue_half", "Verify, then continue (half)", "--^"),
]:
    pts = [(mean_cost(outcomes(name, t)), mean_pass(outcomes(name, t))) for t in THRESHOLDS]
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    ax.plot(xs, ys, style, label=label, markersize=5)
    if name == "hybrid":
        for t, x, y in zip(THRESHOLDS, xs, ys):
            ax.annotate(f"{t}", (x, y), textcoords="offset points", xytext=(4, 6), fontsize=7)

for name, label, marker in [
    ("always_small", "Always small (4B)", "D"),
    ("always_large", "Always large (32B)", "D"),
    ("router", "Route before generation (Jev)", "X"),
]:
    v = outcomes(name)
    lo, hi = ci(v)
    y = mean_pass(v)
    ax.errorbar(mean_cost(v), y, yerr=[[y - lo], [hi - y]], fmt=marker, capsize=3, label=label, markersize=7)


oracle = sum(small[x]["passed"] or large[x]["passed"] for x in tids) / n
ax.axhline(oracle, linestyle=":", color="gray", label=f"Oracle, cold answers ({oracle:.3f})")

ax.set_xlabel("Relative cost per problem (small call = 1)")
ax.set_ylabel(f"{args.benchmark} pass@1")
ax.set_title("Verifying after generation beats routing before it")
ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)
ax.grid(alpha=0.3)
fig.tight_layout()
output = root / "charts" / f"{args.benchmark}_bootstrap_frontier.png"
fig.savefig(output, dpi=200, bbox_inches="tight")
print(f"\nSaved {output}")