"""
benchmark/allocation_comparison.py
====================================
Reproduces the v0.1 → v0.5 policy evolution experiments.

Compares EqualPolicy, RiskOnlyPolicy, ReactivePolicy, SoftmaxPolicy,
and PredictivePolicy across four coupling environments.

Run: python benchmark/allocation_comparison.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import copy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from asas.core.state import make_state
from asas.core.engine import ASASEngine
from asas.core.policy import (EqualPolicy, RiskOnlyPolicy, ReactivePolicy,
                              SoftmaxPolicy, PredictivePolicy, MPCPolicy)
from asas.core.objective import system_entropy, cumulative_entropy

# ── Scenario ───────────────────────────────────────────────────────────────────

SECTORS_INIT = {
    "Hbf":     {"risk": 0.9, "confidence": 0.70},
    "Bridges": {"risk": 0.7, "confidence": 0.20},
    "Messe":   {"risk": 0.4, "confidence": 0.05},
    "Airport": {"risk": 0.6, "confidence": 0.15},
}

BASE_COUPLING = {
    ("Messe",   "Hbf"):     1.0,
    ("Messe",   "Airport"): 1.0,
    ("Bridges", "Airport"): 1.0,
    ("Hbf",     "Bridges"): 1.0,
}

ENVIRONMENTS = {
    "STABLE\n(0.3, 0.08)":    {"scale": 0.3, "decay": 0.08, "color": "#16A34A"},
    "MARGINAL\n(0.5, 0.06)":  {"scale": 0.5, "decay": 0.06, "color": "#2563EB"},
    "UNSTABLE\n(0.7, 0.05)":  {"scale": 0.7, "decay": 0.05, "color": "#D97706"},
    "EXPLOSIVE\n(0.9, 0.03)": {"scale": 0.9, "decay": 0.03, "color": "#DC2626"},
}

POLICIES = {
    "Equal":           EqualPolicy(),
    "Risk-Only":       RiskOnlyPolicy(),
    "Reactive":        ReactivePolicy(gamma=0.0, epsilon=0.05),
    "Softmax γ=0.5":   SoftmaxPolicy(gamma=0.5),
    "Softmax γ=2.0":   SoftmaxPolicy(gamma=2.0),
    "Predictive v0.5": PredictivePolicy(gamma=0.5, outflow_weight=0.5),
    "MPC T=3":         None,   # instantiated per-environment (needs decay param)
    "MPC T=5":         None,
}

STEPS = 60

COLORS = {
    "Equal":           "#D97706",
    "Risk-Only":       "#DC2626",
    "Reactive":        "#7C3AED",
    "Softmax γ=0.5":   "#1D4ED8",
    "Softmax γ=2.0":   "#0891B2",
    "Predictive v0.5": "#059669",
    "MPC T=3":         "#065F46",
    "MPC T=5":         "#022C22",
}


def run_policy(policy, sectors, coupling, steps, decay):
    engine = ASASEngine.from_dict(
        sectors=copy.deepcopy(sectors),
        coupling=coupling,
        policy=policy,
        natural_decay=decay,
        mitigation_strength=0.30,
        learning_rate=0.15,
        forgetting_rate=0.03,
    )
    history = engine.run(steps)
    return [system_entropy(s) for s in history]


MPC_SHARED = dict(
    gamma=0.5, focus_strength=0.5, discount=1.0,
    mitigation_strength=0.30, learning_rate=0.15, forgetting_rate=0.03,
)

def make_policy(name, decay):
    """Instantiate policy, injecting decay for MPC which needs it at init."""
    if name == "MPC T=3":
        return MPCPolicy(horizon=3, natural_decay=decay, **MPC_SHARED)
    if name == "MPC T=5":
        return MPCPolicy(horizon=5, natural_decay=decay, **MPC_SHARED)
    return POLICIES[name]


# ── Run ────────────────────────────────────────────────────────────────────────

print("Running benchmark...")
results = {}
for env_name, cfg in ENVIRONMENTS.items():
    coupling = {k: v * cfg["scale"] for k, v in BASE_COUPLING.items()}
    results[env_name] = {"cfg": cfg}
    for pol_name in POLICIES:
        policy = make_policy(pol_name, cfg["decay"])
        results[env_name][pol_name] = run_policy(
            policy, SECTORS_INIT, coupling, STEPS, cfg["decay"]
        )
    print(f"  {env_name.split(chr(10))[0]} done")

# ── Plot ───────────────────────────────────────────────────────────────────────

iters = list(range(STEPS))
env_names = list(ENVIRONMENTS.keys())

fig = plt.figure(figsize=(18, 12))
gs  = gridspec.GridSpec(2, 4, hspace=0.45, wspace=0.35)

for col, env_name in enumerate(env_names):
    res = results[env_name]
    fc  = res["cfg"]["color"]

    # Row 0: H(t)
    ax = fig.add_subplot(gs[0, col])
    for pol_name, entropy in res.items():
        if pol_name == "cfg":
            continue
        ls = "--" if pol_name == "Equal" else "-"
        ax.plot(iters, entropy, color=COLORS[pol_name], lw=1.8,
                linestyle=ls, label=pol_name)
    ax.set_xlim(0, STEPS-1); ax.set_ylim(bottom=0); ax.grid(alpha=0.2)
    ax.set_title(env_name, fontsize=10, fontweight="bold")
    if col == 0:
        ax.set_ylabel("H(t)", fontsize=9)
        ax.legend(fontsize=7.5)
    for sp in ax.spines.values():
        sp.set_edgecolor(fc); sp.set_linewidth(2.0)

    # Row 1: Cumulative H
    ax = fig.add_subplot(gs[1, col])
    for pol_name, entropy in res.items():
        if pol_name == "cfg":
            continue
        cum = np.cumsum(entropy)
        ls  = "--" if pol_name == "Equal" else "-"
        ax.plot(iters, cum, color=COLORS[pol_name], lw=1.8,
                linestyle=ls, label=f"{pol_name}={cum[-1]:.2f}")
    ax.set_xlim(0, STEPS-1); ax.set_ylim(bottom=0); ax.grid(alpha=0.2)
    ax.legend(fontsize=7)
    if col == 0:
        ax.set_ylabel("Cumulative H", fontsize=9)
    ax.set_xlabel("Step", fontsize=9)
    for sp in ax.spines.values():
        sp.set_edgecolor(fc); sp.set_linewidth(2.0)

fig.suptitle(
    "ASAS Policy Comparison: v0.1 → v0.4\n"
    "EqualPolicy | RiskOnly | Reactive | Softmax(γ=0.5) | Softmax(γ=2.0)",
    fontsize=13, fontweight="bold",
)
plt.savefig("benchmark/policy_comparison.png", dpi=150, bbox_inches="tight")
print("\nSaved: benchmark/policy_comparison.png")

# ── Console summary ────────────────────────────────────────────────────────────

print("\n" + "="*75)
print("Cumulative Entropy Summary (lower = better)")
print("="*75)
pol_names = [p for p in POLICIES]
print(f"{'Env':<22}", end="")
for p in pol_names:
    print(f"  {p:>14}", end="")
print()
print("-"*75)

for env_name in env_names:
    res   = results[env_name]
    label = env_name.replace("\n", " ")
    eq    = sum(res["Equal"])
    print(f"{label:<22}", end="")
    for p in pol_names:
        val = sum(res[p])
        win = "✓" if val < eq and p != "Equal" else " "
        print(f"  {val:>13.2f}{win}", end="")
    print()
print("="*75)
