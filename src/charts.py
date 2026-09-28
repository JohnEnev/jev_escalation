"""Regenerate both benchmark chart pairs from the cleaned JSONs."""
import argparse
import json
import random
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MultipleLocator

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--benchmark", choices=["humaneval", "humanevalplus"], default="humaneval")
args = parser.parse_args()
BENCH = "HumanEval+" if args.benchmark == "humanevalplus" else "HumanEval"
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results" / args.benchmark
OUTPUT = ROOT / "charts"
OUTPUT.mkdir(exist_ok=True)
PREFIX = "humanevalplus_" if args.benchmark == "humanevalplus" else ""
SMALL_COST, LARGE_COST = 1.0, 8.0
THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
ROUTER_THRESHOLDS = [0.5, 0.3, 0.2, 0.1, 0.05, 0.02, 0.01]
HIGHLIGHT_T = 0.6  # fixed in advance on the original HumanEval run; not re-tuned per benchmark

# ---------- data ----------
load = lambda p: {r["task_id"]: r for r in json.loads((DATA / p).read_text())}
small = load("results_small.json")
large = load("results_large.json")
fl = load("results_continuation_first_line.json")
half = load("results_continuation_half.json")
scores = load("results_escalation_scores.json")
router = load("results_router.json")
tids = list(small.keys())
n = len(tids)

failed = {t: not small[t]["passed"] for t in tids}
p_hard = {t: router[t]["jev_probs"].get("hard", 0.0) for t in tids}
p_ok = {t: scores[t]["jev_prob_correct"] for t in tids}


def is_stub(c):
    s = c.strip().lower()
    return s in ("pass", "") or "your code here" in s


def outcomes(strategy, t=None):
    out = []
    for x in tids:
        esc = t is not None and p_ok[x] < t
        if strategy == "always_small":
            out.append((small[x]["passed"], SMALL_COST))
        elif strategy == "always_large":
            out.append((large[x]["passed"], LARGE_COST))
        elif strategy == "route_first":
            to_large = p_hard[x] > t
            out.append((large[x]["passed"], LARGE_COST) if to_large else (small[x]["passed"], SMALL_COST))
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


def point(strategy, t=None):
    v = outcomes(strategy, t)
    return sum(c for _, c in v) / n, sum(p for p, _ in v) / n


def auroc(scores_, labels):
    pos = [s for s, l in zip(scores_, labels) if l]
    neg = [s for s, l in zip(scores_, labels) if not l]
    wins = sum((p > q) + 0.5 * (p == q) for p in pos for q in neg)
    return wins / (len(pos) * len(neg))


# ---------- style ----------
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Inter", "Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#999999",
    "axes.labelcolor": "#333333",
    "xtick.color": "#555555",
    "ytick.color": "#555555",
    "axes.grid": True,
    "axes.grid.axis": "y",
    "grid.color": "#e8e8e8",
    "grid.linewidth": 0.8,
})
ACCENT, BLUE, GREY, DARK, PURPLE = "#d9480f", "#1c7ed6", "#9aa0a6", "#343a40", "#7048e8"


def header(fig, title, subtitle):
    fig.text(0.06, 0.97, title, fontsize=15, fontweight="bold", color="#111111", ha="left", va="top")
    fig.text(0.06, 0.915, subtitle, fontsize=10, color="#555555", ha="left", va="top", linespacing=1.4)


def footer(fig, text):
    fig.text(0.06, 0.02, text, fontsize=8, color="#888888", ha="left")


# ---------- chart 1: frontier ----------
fig, ax = plt.subplots(figsize=(9, 6))
fig.subplots_adjust(left=0.09, right=0.95, top=0.80, bottom=0.14)

curves = {}
for name, color, ls, lw, ms, ths in [
    ("route_first", PURPLE, ":", 1.8, 4, ROUTER_THRESHOLDS),
    ("continue_half", GREY, "--", 1.6, 4, THRESHOLDS),
    ("cascade", BLUE, "-", 1.8, 4, THRESHOLDS),
    ("hybrid", ACCENT, "-", 2.8, 5, THRESHOLDS),
]:
    pts = [point(name, t) for t in ths]
    curves[name] = pts
    ax.plot([x for x, _ in pts], [y for _, y in pts], ls, color=color, lw=lw,
            marker="o", ms=ms, zorder=3 if name == "hybrid" else 2)

print(f"Route-first sweep on {BENCH} (threshold, cost, pass@1):")
for t, (c, p) in zip(ROUTER_THRESHOLDS, curves["route_first"]):
    print(f"  {t:<5} {c:5.2f}  {p:.3f}")

# direct labels
i7, i8 = THRESHOLDS.index(0.7), THRESHOLDS.index(0.8)
x, y = curves["hybrid"][i7]
ax.annotate("Verify, restart, retry stubs", (x, y), xytext=(10, 12), textcoords="offset points",
            ha="left", color=ACCENT, fontsize=9.5, fontweight="bold")
x, y = curves["cascade"][i8]
ax.annotate("Verify, then restart", (x, y), xytext=(-10, -6), textcoords="offset points",
            ha="right", color=BLUE, fontsize=9.5)
x, y = curves["continue_half"][i8]
ax.annotate("Verify, then continue", (x, y), xytext=(0, -18), textcoords="offset points",
            ha="center", color=GREY, fontsize=9.5)
x, y = curves["route_first"][-1]
ax.annotate("Route first (Jev),\nswept threshold", (x, y), xytext=(8, -4), textcoords="offset points",
            ha="left", va="top", color=PURPLE, fontsize=9.5)

# highlighted hybrid point, annotation pinned to the top-left of the plot
bx, by = curves["hybrid"][THRESHOLDS.index(HIGHLIGHT_T)]
lx, ly = point("always_large")
ax.scatter([bx], [by], s=110, facecolor="white", edgecolor=ACCENT, lw=2.2, zorder=5)
ax.annotate(
    f"{by:.1%} solved at {bx:.1f}× cost (threshold {HIGHLIGHT_T})\n"
    f"+{(by - ly) * 100:.0f} pts vs always-32B, at {bx / LARGE_COST:.0%} of its cost",
    (bx, by), xytext=(0.02, 0.985), textcoords="axes fraction", va="top", ha="left",
    fontsize=9, color=DARK, arrowprops=dict(arrowstyle="-", color=DARK, lw=0.8),
)

# baselines
sx, sy = point("always_small")
ax.scatter(sx, sy, s=60, color=DARK, zorder=4)
ax.annotate("Always 4B", (sx, sy), xytext=(0, -16), textcoords="offset points", ha="center", fontsize=9, color=DARK)
ax.scatter(lx, ly, s=60, color=DARK, zorder=4)
ax.annotate("Always 32B", (lx, ly), xytext=(-8, -15), textcoords="offset points", ha="right", fontsize=9, color=DARK)

# oracle
oracle = sum(small[x]["passed"] or large[x]["passed"] for x in tids) / n
ax.axhline(oracle, color=GREY, ls=":", lw=1.2, zorder=1)
ax.text(9.75, oracle + 0.004, f"Oracle, best of 4B/32B per problem: {oracle:.1%}",
        ha="right", fontsize=8.5, color="#777777")

# axes follow the data, so the same script works for both benchmarks
all_y = [y for pts in curves.values() for _, y in pts] + [sy, ly, oracle]
ax.set_xlim(0.5, 9.8)
ax.set_ylim(min(all_y) - 0.03, max(all_y) + 0.045)
ax.set_xticks([1, 2, 4, 6, 8])
ax.yaxis.set_major_locator(MultipleLocator(0.05))
ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}×"))
ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0%}"))
ax.set_xlabel("Cost per problem (one 4B call = 1×, one 32B call = 8×)")
ax.set_ylabel(f"{BENCH} problems solved")

header(fig, "Checking the answer beats guessing the difficulty",
       "Qwen3-4B answers first; Jev decides whether to hand the problem to Qwen3-32B.\n"
       "Each point on a curve is a different escalation threshold.")
footer(fig, f"{n} {BENCH} problems, pass@1. Cost is a proxy based on parameter count. "
            "Differences tested with paired bootstrap (see table).")
fig.savefig(OUTPUT / f"{PREFIX}chart_frontier.png", dpi=220)
fig.savefig(OUTPUT / f"{PREFIX}chart_frontier.svg")

# ---------- chart 2: why verification works ----------
random.seed(1)
au_router = auroc([p_hard[t] for t in tids], [failed[t] for t in tids])
au_verify = auroc([1 - p_ok[t] for t in tids], [failed[t] for t in tids])

fig, axes = plt.subplots(1, 2, figsize=(9, 4.8), sharey=True)
fig.subplots_adjust(left=0.13, right=0.97, top=0.72, bottom=0.17, wspace=0.12)

panels = [
    (axes[0], p_hard, "Before generation", "Jev reads the prompt: p(hard)", au_router),
    (axes[1], p_ok, "After generation", "Jev reads the 4B answer: p(correct)", au_verify),
]
for ax, vals, title, xlabel, au in panels:
    for row, (is_fail, color) in enumerate([(False, GREY), (True, ACCENT)]):
        xs = [vals[t] for t in tids if failed[t] == is_fail]
        ys = [row + random.uniform(-0.2, 0.2) for _ in xs]
        ax.scatter(xs, ys, s=18, color=color, alpha=0.75, edgecolor="none")
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.55, 1.55)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["4B answer\npassed", "4B answer\nfailed"])
    ax.set_xlabel(xlabel)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color="#eeeeee")
    ax.set_title(f"{title}   ·   AUROC {au:.2f}", loc="left", fontsize=11, fontweight="bold", color=DARK)

axes[0].text(0.3, 0.5, "Scores bunch near 0, so the default\n0.5 cutoff almost never picks 32B",
             fontsize=8, color=DARK, va="center")
axes[1].axvline(HIGHLIGHT_T, color=DARK, lw=0.8, ls="--")
axes[1].text(HIGHLIGHT_T - 0.015, 1.45, f"escalate below {HIGHLIGHT_T}", ha="right", fontsize=8, color=DARK)

header(fig, "Same model, different question",
       f"Each dot is one {BENCH} problem; orange ones are problems the 4B model got wrong.\n"
       "From the prompt, Jev's signal is weak and squeezed near zero. From the answer, it's strong.")
footer(fig, "AUROC: probability that a random failed problem is ranked as riskier than a random passed one (0.5 = chance).")
fig.savefig(OUTPUT / f"{PREFIX}chart_verification.png", dpi=220)
fig.savefig(OUTPUT / f"{PREFIX}chart_verification.svg")

print(f"Saved {PREFIX}chart_frontier and {PREFIX}chart_verification PNG/SVG for {BENCH}")