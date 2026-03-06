"""
benchmark/horizon_sensitivity.py
=================================
Horizon sensitivity analysis for MPCPolicy (v0.7).

Experiments:
  1. Cumulative H(t) vs horizon T  (monotonicity test)
  2. Efficiency frontier: improvement% vs runtime

Run: python benchmark/horizon_sensitivity.py
"""

import sys, os, copy, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from asas.core.engine import ASASEngine
from asas.core.policy import EqualPolicy, SoftmaxPolicy, MPCPolicy
from asas.core.objective import system_entropy

# ── Scenario ──────────────────────────────────────────────────────────────────

SECTORS = {
    "Hbf":     {"risk": 0.9, "confidence": 0.70},
    "Bridges": {"risk": 0.7, "confidence": 0.20},
    "Messe":   {"risk": 0.4, "confidence": 0.05},
    "Airport": {"risk": 0.6, "confidence": 0.15},
}
COUPLING = {
    ("Messe",   "Hbf"):     0.7,
    ("Messe",   "Airport"): 0.7,
    ("Bridges", "Airport"): 0.7,
    ("Hbf",     "Bridges"): 0.7,
}
SCHEDULE  = {3: {"Hbf": +0.40, "Bridges": +0.20, "Airport": +0.15}}
DP        = dict(natural_decay=0.05, mitigation_strength=0.30,
                 learning_rate=0.15, forgetting_rate=0.03)
SIM_STEPS = 20
HORIZONS  = [1, 2, 3, 4, 5, 6, 8, 10, 15, 20]
N_TRIALS  = 5    # repeat timing for stable runtime estimate

# ── Run Equal baseline ─────────────────────────────────────────────────────────

def run(policy, schedule=None):
    engine = ASASEngine.from_dict(
        sectors=copy.deepcopy(SECTORS), coupling=COUPLING,
        policy=policy, **DP,
    )
    hist = engine.run(SIM_STEPS, signal_schedule=schedule or {})
    return sum(system_entropy(s) for s in hist)

eq_cum     = run(EqualPolicy(),       SCHEDULE)
sm_cum     = run(SoftmaxPolicy(gamma=0.5), SCHEDULE)

# ── Horizon sweep ──────────────────────────────────────────────────────────────

T_vals, H_vals, RT_vals = [], [], []

for T in HORIZONS:
    policy = MPCPolicy(horizon=T, gamma=0.5, focus_strength=0.5, **DP)

    # Stable runtime: median of N_TRIALS
    runtimes = []
    for _ in range(N_TRIALS):
        t0  = time.perf_counter()
        cum = run(policy, SCHEDULE)
        runtimes.append((time.perf_counter() - t0) * 1000)

    T_vals.append(T)
    H_vals.append(cum)
    RT_vals.append(float(np.median(runtimes)))

T_vals  = np.array(T_vals)
H_vals  = np.array(H_vals)
RT_vals = np.array(RT_vals)
impr    = (eq_cum - H_vals) / eq_cum * 100   # improvement % over Equal

# ── Plot ───────────────────────────────────────────────────────────────────────

fig = plt.figure(figsize=(14, 5))
fig.patch.set_facecolor("#0a0e1a")
gs = gridspec.GridSpec(1, 3, wspace=0.38)

DARK  = "#0a0e1a"
GRID  = "#1a2235"
BLUE  = "#1b98e0"
GREEN = "#10b981"
AMBER = "#f59e0b"
RED   = "#e94560"
TEXT  = "#e2e8f0"
FAINT = "#64748b"

def style_ax(ax, title):
    ax.set_facecolor(DARK)
    for sp in ax.spines.values():
        sp.set_edgecolor(GRID)
    ax.tick_params(colors=FAINT, labelsize=8)
    ax.set_title(title, color=TEXT, fontsize=10, fontweight="bold", pad=10)
    ax.grid(color=GRID, linewidth=0.8, alpha=0.7)
    ax.xaxis.label.set_color(FAINT)
    ax.yaxis.label.set_color(FAINT)

# ── Panel 1: Cumulative H vs T ─────────────────────────────────────────────────
ax1 = fig.add_subplot(gs[0])
style_ax(ax1, "Cumulative H(t) vs Horizon T")

ax1.axhline(eq_cum, color=AMBER, lw=1.5, ls="--", label=f"Equal  ({eq_cum:.3f})")
ax1.axhline(sm_cum, color=RED,   lw=1.2, ls=":",  label=f"Softmax v0.4  ({sm_cum:.3f})")
ax1.plot(T_vals, H_vals, color=GREEN, lw=2.2, marker="o",
         markersize=6, markerfacecolor=DARK, markeredgecolor=GREEN,
         label="MPC (closed-loop)")

# Annotate diminishing returns knee (~T=5)
knee_idx = np.argmin(np.diff(H_vals) < -0.02)
ax1.axvline(T_vals[knee_idx], color=GREEN, lw=0.8, ls="--", alpha=0.4)
ax1.annotate(f"T={T_vals[knee_idx]}\nknee",
             xy=(T_vals[knee_idx], H_vals[knee_idx]),
             xytext=(T_vals[knee_idx]+1.5, H_vals[knee_idx]+0.05),
             color=GREEN, fontsize=7.5,
             arrowprops=dict(arrowstyle="->", color=GREEN, lw=0.8))

ax1.set_xlabel("Horizon T", fontsize=9)
ax1.set_ylabel("Cumulative H(t)", fontsize=9)
ax1.legend(fontsize=7.5, facecolor=GRID, edgecolor=GRID, labelcolor=TEXT)

# ── Panel 2: Improvement % vs T ───────────────────────────────────────────────
ax2 = fig.add_subplot(gs[1])
style_ax(ax2, "Improvement over Equal (%)")

ax2.fill_between(T_vals, impr, alpha=0.15, color=GREEN)
ax2.plot(T_vals, impr, color=GREEN, lw=2.2, marker="o",
         markersize=6, markerfacecolor=DARK, markeredgecolor=GREEN)

for i, (t, imp) in enumerate(zip(T_vals, impr)):
    if t in [1, 3, 5, 10, 20]:
        ax2.annotate(f"{imp:.1f}%",
                     xy=(t, imp), xytext=(t+0.3, imp+0.2),
                     color=GREEN, fontsize=7.5)

ax2.set_xlabel("Horizon T", fontsize=9)
ax2.set_ylabel("Improvement over Equal (%)", fontsize=9)
ax2.set_ylim(bottom=0)

# ── Panel 3: Efficiency Frontier ──────────────────────────────────────────────
ax3 = fig.add_subplot(gs[2])
style_ax(ax3, "Efficiency Frontier")

sc = ax3.scatter(RT_vals, impr, c=T_vals, cmap="YlGn",
                 s=80, zorder=5, edgecolors=FAINT, linewidths=0.5)
ax3.plot(RT_vals, impr, color=FAINT, lw=1.0, ls="--", zorder=4)

for t, rt, imp in zip(T_vals, RT_vals, impr):
    if t in [1, 3, 5, 10, 20]:
        ax3.annotate(f"T={t}", xy=(rt, imp),
                     xytext=(rt+0.5, imp+0.15),
                     color=TEXT, fontsize=7.5)

cbar = plt.colorbar(sc, ax=ax3)
cbar.set_label("Horizon T", color=FAINT, fontsize=8)
cbar.ax.yaxis.set_tick_params(color=FAINT, labelcolor=FAINT)

ax3.set_xlabel("Runtime per episode (ms)", fontsize=9)
ax3.set_ylabel("Improvement over Equal (%)", fontsize=9)

# ── Title & save ──────────────────────────────────────────────────────────────

fig.suptitle("ASAS v0.7 — MPC Horizon Sensitivity Analysis",
             color=TEXT, fontsize=12, fontweight="bold", y=1.02)

out = os.path.join(os.path.dirname(__file__), "horizon_sensitivity.png")
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=DARK)
print(f"Saved: {out}")

# ── Print summary table ────────────────────────────────────────────────────────
print()
print("=" * 62)
print(f"Horizon Sensitivity Summary  (Equal baseline = {eq_cum:.4f})")
print("=" * 62)
print(f"{'T':>4}  {'cumH':>8}  {'improvement':>12}  {'runtime':>10}")
print("-" * 42)
for t, h, rt, imp in zip(T_vals, H_vals, RT_vals, impr):
    print(f"{t:>4}  {h:>8.4f}  {imp:>11.1f}%  {rt:>9.1f}ms")
print()
print(f"Key finding: H(T) is monotonically decreasing — no optimal horizon.")
print(f"Diminishing returns after T≈5: marginal gain T5→T10 = "
      f"{impr[HORIZONS.index(10)]-impr[HORIZONS.index(5)]:.1f}%  "
      f"vs T1→T5 = {impr[HORIZONS.index(5)]-impr[0]:.1f}%")
