"""
asas/core/dynamics.py
=====================
State transition equations for ASAS.

This module contains only pure functions:
    state × allocation × signals → new_state

No policy logic. No objective function. No LLM calls.

Equations
---------
Risk evolution (systemic layer):
    r_i(t+1) = r_i(t)·(1-δ)                           natural decay
             + Σ_{j→i} w_ji·(1-c_j(t))²               spillover
             - μ·a_i(t)                                 mitigation
             + external_i(t)                            event injection
    clipped to [0,1]

Confidence evolution (epistemic layer):
    c_i(t+1) = c_i(t) + η·a_i(t)·(1-c_i(t))          bounded learning
                       - ρ·c_i(t)                       forgetting
    clipped to [0,1]

    Fixed point: c_i* = η·a_i* / (ρ + η·a_i*) ∈ (0,1)
    Forgetting (ρ>0) is structurally necessary: without it c_i→1,
    u_i→0, and the system loses its ability to differentiate sectors.

Spillover term (1-c_j)²:
    Unknown sectors are unpredictable — unpredictability propagates.
    c_j=0.05 → spillover ∝ 0.90  (barely monitored)
    c_j=0.90 → spillover ∝ 0.01  (well-understood)

External event injection:
    The perception layer supplies external_i(t) ∈ ℝ.
    Positive = risk increase (e.g. strike alert at Hbf: +0.4)
    Negative = risk decrease (e.g. incident resolved)
    This is how open-world events enter the mathematical framework.
"""

from __future__ import annotations
import copy
from typing import Dict, Optional, Tuple

from .state import SystemState, SectorState


# ─────────────────────────────────────────────
# Network measures (used by policy and dynamics)
# ─────────────────────────────────────────────

def outflow(state: SystemState) -> Dict[str, float]:
    """
    Spillover a sector is currently generating toward the network.

        outflow_i = Σ_{i→j} w_ij · r_i · (1 - c_i)²

    Both terms required:
        r_i:        low-risk sector generates little spillover even if unmonitored
        (1-c_i)²:  unmonitored sector propagates unpredictably

    Semantics: if attention raises c_i, how much network-wide spillover
    is prevented? This is the marginal system value of attending to i.
    Points to SOURCES. Self-correcting as c_i rises.

    Explicit Euler consistency:
        Uses r_i(t) and c_i(t) — the same previous-step values used by
        dynamics.step() for spillover computation. The outflow signal seen
        by the policy therefore exactly matches what step() will propagate,
        making the policy's reasoning grounded in the actual dynamics.
    """
    result = {sid: 0.0 for sid in state.sector_ids}
    for (src, tgt), w in state.coupling.items():
        s = state.sectors[src]
        result[src] += w * s.risk * s.uncertainty ** 2
    return result


def inflow(state: SystemState) -> Dict[str, float]:
    """
    Spillover arriving at each sector from upstream neighbors.

        inflow_i = Σ_{j→i} w_ji · r_j(t) · (1 - c_j(t))²

    Diagnostic use only — this is the quantity actually arriving at each
    sector per step, matching the physical spillover in dynamics.step().

    Why not used as a priority signal:
        inflow is highest at structural sinks (Airport receives spillover
        from Messe + Bridges). Attending the sink does not reduce the
        spillover flowing into it — the fix must come upstream.
        See v0.2 failure analysis in docs/theory.md.

    Physical consistency:
        Uses the same r_src · u_src² formula as dynamics.step() and
        outflow(). All three functions now agree on what "spillover" means.
        A diagnostic built on inflow() reflects the true load on each sector.
    """
    result = {sid: 0.0 for sid in state.sector_ids}
    for (src, tgt), w in state.coupling.items():
        s = state.sectors[src]
        result[tgt] += w * s.risk * s.uncertainty ** 2
    return result


def centrality(state: SystemState) -> Dict[str, float]:
    """
    Normalized weighted in-degree.
    High centrality = structural risk sink.
    Used as penalty: sinks don't benefit from local attention alone.
    """
    raw = {sid: 0.0 for sid in state.sector_ids}
    for (src, tgt), w in state.coupling.items():
        raw[tgt] += w
    max_raw = max(raw.values()) if any(v > 0 for v in raw.values()) else 1.0
    return {sid: raw[sid] / max_raw for sid in state.sector_ids}


# ─────────────────────────────────────────────
# State transition
# ─────────────────────────────────────────────

def step(
    state: SystemState,
    allocation: Dict[str, float],
    external_signals: Optional[Dict[str, float]] = None,
    natural_decay:       float = 0.05,
    mitigation_strength: float = 0.30,
    learning_rate:       float = 0.15,
    forgetting_rate:     float = 0.03,
) -> SystemState:
    """
    Advance the network by one time step.

    Parameters
    ----------
    state            : current SystemState
    allocation       : a_i values from the policy (must sum to 1)
    external_signals : risk deltas from perception layer
                       e.g. {"Hbf": +0.4} on strike day
    natural_decay    : δ, fraction of risk dissipating per step
    mitigation_strength: μ, risk reduction per unit attention
    learning_rate    : η, confidence gain per unit attention per step
    forgetting_rate  : ρ, structural confidence decay per step

    Returns
    -------
    New SystemState (immutable — original state is never modified)
    """
    ext = external_signals or {}

    # ── Risk evolution ─────────────────────────────────────────
    new_risk: Dict[str, float] = {}

    for sid in state.sector_ids:
        s = state.sectors[sid]
        r = s.risk * (1.0 - natural_decay)          # natural decay
        r -= mitigation_strength * allocation[sid]   # mitigation
        r += ext.get(sid, 0.0)                       # event injection
        new_risk[sid] = r

    # Spillover: risk × uncertainty² propagates from source to target.
    #
    #   spillover_{src→tgt} = w · r_src(t) · u_src(t)²
    #
    # Both terms are necessary:
    #   r_src: a low-risk sector generates little spillover even if unmonitored
    #   u_src²: an unmonitored sector propagates more unpredictably
    #
    # Explicit Euler stepping — important for scientific reproducibility:
    #   Spillover uses r_src(t) and u_src(t), values from the CURRENT step
    #   before any updates are applied. This is explicit (forward) Euler
    #   integration: new_risk is computed entirely from the previous state.
    #   A sector's rising risk does not immediately amplify its own spillover
    #   within the same step, preventing implicit feedback amplification
    #   and making the dynamics analytically tractable.
    #
    # Original (v0.3): w · u²      — ignores source risk, theoretically unsound
    # Fixed  (v0.4+):  w · r · u²  — physically correct, consistent with outflow()
    for (src, tgt), w in state.coupling.items():
        s_src = state.sectors[src]
        new_risk[tgt] += w * s_src.risk * s_src.uncertainty ** 2

    # Clip systemic layer to [0,1]
    new_risk = {sid: min(max(v, 0.0), 1.0) for sid, v in new_risk.items()}

    # ── Confidence evolution ───────────────────────────────────
    new_conf: Dict[str, float] = {}

    for sid in state.sector_ids:
        s   = state.sectors[sid]
        c   = s.confidence
        a   = allocation[sid]
        dc  = learning_rate * a * (1.0 - c) - forgetting_rate * c
        new_conf[sid] = min(max(c + dc, 0.0), 1.0)

    # ── Assemble new state ─────────────────────────────────────
    new_sectors = {
        sid: SectorState(
            sector_id=sid,
            risk=new_risk[sid],
            confidence=new_conf[sid],
            allocation=allocation[sid],
        )
        for sid in state.sector_ids
    }

    return SystemState(
        sectors=new_sectors,
        coupling=state.coupling,
        step=state.step + 1,
    )
