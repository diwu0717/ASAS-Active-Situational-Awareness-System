"""
asas/core/objective.py
======================
Objective functions and system health metrics.

Primary objective: System Entropy H(t)
---------------------------------------
    H(t) = Σ_i  a_i(t) · r_i(t) · u_i(t)

Measures risk-weighted residual uncertainty under current attention.
    H high → attention directed at sectors we don't understand
    H low  → confidence built where risk is highest

H(t) is the single number that characterizes how well the system is
doing at its core task: directing attention to where risk is high
and understanding is low.

Theoretical result (v0.1–v0.4 experiments)
-------------------------------------------
Under persistent symmetric spillover with linear confidence dynamics,
the reactive policy minimizing H(t) converges to uniform allocation.
Uniform allocation is the γ→∞ limit of the entropy-regularized policy.

This means: ASAS value emerges when external event signals break
network symmetry, shifting the optimal allocation away from uniform.
H(t) measures whether that shift is justified.

Secondary metrics
-----------------
H_alloc(a)  allocation entropy  — measures concentration of attention
             H_alloc = log(N) for uniform; 0 for full concentration
             Used to diagnose Airport attractor behavior.

Trend        sign(ΔH / Δt)      — rising/stable/falling
"""

from __future__ import annotations
import math
from typing import Dict, List, Optional

from .state import SystemState


# ─────────────────────────────────────────────
# Primary objective
# ─────────────────────────────────────────────

def system_entropy(state: SystemState) -> float:
    """
    H(t) = Σ_i  a_i · r_i · u_i

    Primary ASAS health metric. Lower is better.
    """
    return sum(s.entropy_contribution for s in state.sectors.values())


def sector_entropy_contributions(state: SystemState) -> Dict[str, float]:
    """Per-sector breakdown of H(t) = Σ h_i."""
    return {sid: s.entropy_contribution for sid, s in state.sectors.items()}


# ─────────────────────────────────────────────
# Secondary metrics
# ─────────────────────────────────────────────

def allocation_entropy(state: SystemState) -> float:
    """
    H(a) = -Σ_i  a_i · log(a_i)

    Measures spread of attention allocation.
    Maximum = log(N) for uniform distribution.
    Minimum = 0 for full concentration on one sector.

    Diagnostic use: if H(a) << log(N), allocation is concentrated.
    Compare with system_entropy to distinguish:
        Low H(t) + high H(a) → good: risk reduced, attention spread
        Low H(t) + low H(a)  → fragile: concentrated on one sector
        High H(t) + any H(a) → bad: risk not being managed
    """
    h = 0.0
    for s in state.sectors.values():
        if s.allocation > 1e-10:
            h -= s.allocation * math.log(s.allocation)
    return h


def max_allocation_entropy(state: SystemState) -> float:
    """log(N) — maximum possible H(a), achieved by uniform allocation."""
    return math.log(state.N)


def entropy_trend(history: List[SystemState], window: int = 3) -> str:
    """
    'rising', 'falling', or 'stable' based on recent H(t) history.

    Parameters
    ----------
    history : list of past SystemState (oldest first)
    window  : number of recent steps to consider
    """
    if len(history) < window:
        return "unknown"
    recent = [system_entropy(s) for s in history[-window:]]
    delta  = recent[-1] - recent[0]
    if delta >  0.01:
        return "rising"
    if delta < -0.01:
        return "falling"
    return "stable"


def cumulative_entropy(history: List[SystemState]) -> float:
    """Σ H(t) over all steps. Used in benchmark comparisons."""
    return sum(system_entropy(s) for s in history)


# ─────────────────────────────────────────────
# Structured status report
# ─────────────────────────────────────────────

def status_report(
    state: SystemState,
    history: Optional[List[SystemState]] = None,
    top_n_hotspots: int = 3,
) -> Dict:
    """
    Structured status report for cognitive hub consumption.

    Contains everything an LLM needs to generate a situation report:
    current H(t), sector states, priority allocations, hotspots, trend.

    Parameters
    ----------
    state            : current SystemState
    history          : past states for trend computation
    top_n_hotspots   : number of top-scoring sectors to highlight
    """
    h_sys   = system_entropy(state)
    h_alloc = allocation_entropy(state)
    h_max   = max_allocation_entropy(state)
    trend   = entropy_trend(history or [])

    # Priority allocations — sorted highest first
    allocs = sorted(
        [(sid, s.allocation) for sid, s in state.sectors.items()],
        key=lambda x: -x[1],
    )

    # Uncertainty hotspots — highest r·u score
    hotspots = sorted(
        [(sid, s.base_score) for sid, s in state.sectors.items()],
        key=lambda x: -x[1],
    )[:top_n_hotspots]

    return {
        "step":             state.step,
        "global_entropy":   round(h_sys, 4),
        "allocation_entropy": round(h_alloc, 4),
        "entropy_utilization": round(h_alloc / h_max, 3) if h_max > 0 else 0,
        "entropy_trend":    trend,
        "sector_states":    state.to_dict()["sectors"],
        "priority_allocations": [
            {"sector": sid, "allocation_pct": round(a * 100, 1)}
            for sid, a in allocs
        ],
        "uncertainty_hotspots": [
            {"sector": sid, "score": round(score, 3)}
            for sid, score in hotspots
        ],
    }
