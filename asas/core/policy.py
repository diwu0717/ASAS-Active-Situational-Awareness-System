"""
asas/core/policy.py
===================
Allocation policies: the pluggable strategy layer.

Design
------
All policies implement one method:

    allocate(state: SystemState) -> Dict[str, float]

Returns a_i values summing to 1. The engine calls this once per step.
Policies are stateless — all information comes from SystemState.

This makes policies:
    - Testable in isolation
    - Swappable at runtime
    - Comparable in benchmarks (same engine, different policy)

Policy hierarchy (mirrors ASAS version history)
------------------------------------------------
EqualPolicy       v∞  Uniform baseline. γ→∞ limit of SoftmaxPolicy.
                      Optimal reactive strategy under symmetric spillover.
RiskOnlyPolicy    v0  Risk-proportional. γ→0 of SoftmaxPolicy (no outflow).
                      Ignores epistemic state entirely.
ReactivePolicy    v0.1 Score = r·u. Adds uncertainty awareness.
SoftmaxPolicy     v0.4 Score via softmax with temperature γ.
                      Unifies all reactive policies: γ controls concentration.
                      γ→0: greedy. γ→∞: uniform.

Extending
---------
To add a new policy (e.g. one-step predictive):

    class PredictivePolicy(AllocationPolicy):
        def allocate(self, state: SystemState) -> Dict[str, float]:
            # Estimate ΔH from one-step rollout for each sector
            ...

Then pass it to ASASEngine:
    engine = ASASEngine(state, policy=PredictivePolicy(), ...)
"""

from __future__ import annotations
import math
from abc import ABC, abstractmethod
from typing import Dict

from .state import SystemState
from .dynamics import outflow, centrality


# ─────────────────────────────────────────────
# Abstract base
# ─────────────────────────────────────────────

class AllocationPolicy(ABC):
    """
    Abstract base class for all ASAS allocation policies.
    Subclasses implement allocate() and nothing else.
    """

    @abstractmethod
    def allocate(self, state: SystemState) -> Dict[str, float]:
        """
        Compute attention allocation for the current state.

        Parameters
        ----------
        state : current SystemState (read-only)

        Returns
        -------
        dict of sector_id → allocation fraction, summing to 1.0
        """

    def __repr__(self) -> str:
        return self.__class__.__name__

    # ── Shared utility ─────────────────────────────────────────

    @staticmethod
    def _normalize(scores: Dict[str, float]) -> Dict[str, float]:
        total = sum(scores.values())
        if total < 1e-10:
            n = len(scores)
            return {k: 1.0 / n for k in scores}
        return {k: v / total for k, v in scores.items()}

    @staticmethod
    def _softmax(scores: Dict[str, float], gamma: float) -> Dict[str, float]:
        """Numerically stable softmax: a_i ∝ exp(score_i / γ)."""
        max_s = max(scores.values())
        exp_s = {k: math.exp((v - max_s) / gamma) for k, v in scores.items()}
        return AllocationPolicy._normalize(exp_s)


# ─────────────────────────────────────────────
# Concrete policies
# ─────────────────────────────────────────────

class EqualPolicy(AllocationPolicy):
    """
    Uniform allocation: a_i = 1/N for all i.

    Theoretical note:
        This is the γ→∞ limit of SoftmaxPolicy.
        Under persistent symmetric spillover with linear dynamics,
        this is the optimal reactive strategy (proven empirically
        via γ-scan in benchmark/allocation_comparison.py).

        ASAS value over Equal emerges only when external event signals
        break network symmetry — making some sectors genuinely more
        important than others on a given day.
    """

    def allocate(self, state: SystemState) -> Dict[str, float]:
        n = state.N
        return {sid: 1.0 / n for sid in state.sector_ids}


class RiskOnlyPolicy(AllocationPolicy):
    """
    Risk-proportional allocation: a_i ∝ r_i.

    Ignores epistemic state (confidence/uncertainty) entirely.
    Equivalent to SoftmaxPolicy with γ→0 and outflow_weight=0.
    Tends to over-allocate to high-risk sinks (Airport attractor).
    """

    def allocate(self, state: SystemState) -> Dict[str, float]:
        return self._normalize(state.risk_vector())


class ReactivePolicy(AllocationPolicy):
    """
    Score = r_i · u_i (risk AND uncertainty).
    With softmax temperature γ and exploration floor ε.

    The multiplicative structure encodes AND logic:
    both risk and uncertainty must be high to warrant attention.

    Parameters
    ----------
    gamma   : softmax temperature (default: greedy normalize)
    epsilon : exploration floor — minimum attention per sector
    """

    def __init__(self, gamma: float = 0.0, epsilon: float = 0.05):
        self.gamma   = gamma
        self.epsilon = epsilon

    def allocate(self, state: SystemState) -> Dict[str, float]:
        scores = state.score_vector()   # r·u
        n = state.N

        if self.gamma > 0:
            base = self._softmax(scores, self.gamma)
        else:
            base = self._normalize({k: max(v, 0.0) for k, v in scores.items()})

        if self.epsilon > 0:
            uniform = {sid: 1.0 / n for sid in state.sector_ids}
            return {
                sid: (1 - self.epsilon) * base[sid] + self.epsilon * uniform[sid]
                for sid in state.sector_ids
            }
        return base


class SoftmaxPolicy(AllocationPolicy):
    """
    ASAS v0.4: Entropy-regularized softmax allocation.

    Derived from the optimization problem:
        min_a  Σ a_i·score_i  -  γ·H_alloc(a)
        s.t.   Σ a_i = 1, a_i ≥ 0

    Solution:  a_i* ∝ exp(score_i / γ)

    Score:  score_i = r_i·u_i + outflow_weight·outflow_i

    The outflow term shifts priority toward network sources (low-confidence
    sectors with many downstream neighbors) rather than risk sinks.

    γ controls the concentration-spread tradeoff:
        γ → 0  : greedy (concentrated on argmax score)
        γ → ∞  : uniform (Equal policy)
        γ ∈ [0.3, 1.5] : balanced, recommended range

    No epsilon floor needed: softmax guarantees a_i > 0 always.

    Parameters
    ----------
    gamma          : temperature / regularization strength
    outflow_weight : weight of network-source signal in score (0 = pure r·u)
    """

    def __init__(self, gamma: float = 0.5, outflow_weight: float = 0.5):
        self.gamma          = gamma
        self.outflow_weight = outflow_weight

    def allocate(self, state: SystemState) -> Dict[str, float]:
        of = outflow(state)
        scores = {
            sid: s.base_score + self.outflow_weight * of[sid]
            for sid, s in state.sectors.items()
        }
        return self._softmax(scores, self.gamma)

    def __repr__(self) -> str:
        return f"SoftmaxPolicy(γ={self.gamma}, λ={self.outflow_weight})"


class AdaptivePolicy(AllocationPolicy):
    """
    Extension point: policy with mutable internal state.

    Useful for:
        - Adaptive γ (adjusts based on entropy trend)
        - One-step predictive (rolls out dynamics to estimate ΔH)
        - History-aware allocation

    Subclass this and implement allocate() + optionally update().
    """

    def update(self, state: SystemState) -> None:
        """Called after each step. Override to update internal state."""

    @abstractmethod
    def allocate(self, state: SystemState) -> Dict[str, float]:
        ...


class PredictivePolicy(AdaptivePolicy):
    """
    ASAS v0.5 — One-step predictive allocation.

    Motivation
    ----------
    Reactive policies (v0.1–v0.4) optimize the current score r_i·u_i.
    This is a greedy signal: it answers "where is uncertainty highest now?"
    not "where will attention reduce H most over the next step?"

    In symmetric networks these questions have the same answer (uniform).
    In asymmetric networks — created by external event injection — they
    diverge. This divergence is where PredictivePolicy gains advantage.

    Derivation of ΔH_i
    -------------------
    We want: ∂H(t+1)/∂a_i — the marginal H reduction from one unit of
    attention directed to sector i.

    Three causal paths contribute:

    Path 1 — Direct allocation effect:
        ∂H/∂a_i|direct = r_i · u_i
        (more attention to i increases its weight in H — offset by paths 2,3)

    Path 2 — Confidence effect (local):
        Attention raises c_i → lowers u_i → lowers H contribution of i
        ∂c_i/∂a_i       = η·(1 - c_i)        [from confidence dynamics]
        ∂H/∂a_i|conf    = -η·(1-c_i)·a_i·r_i

    Path 3 — Spillover prevention (network):
        Attention raises c_i → lowers u_i → reduces r·u² spillover to neighbors
        ∂r_j/∂a_i       = w_ij · r_i · 2u_i · η·(1-c_i)
        ∂H/∂a_i|spill   = -2η·λ·r_i·u_i·(1-c_i)·Σ_j w_ij

    Combined predictive score:
        ΔH_i = r_i·u_i·[1 - η·(1-c_i)·a_i/u_i - 2η·λ·(1-c_i)·Σ_j w_ij]

    Key insight:
        The term 2η·λ·(1-c_i)·Σ_j w_ij is asymmetric across sectors
        even in a symmetric network IF initial confidence values differ.
        This is the mechanism by which PredictivePolicy breaks the
        symmetry invariance that constrains all reactive policies.

    Compared to SoftmaxPolicy(v0.4):
        v0.4 score: r_i·u_i + λ·outflow_i
        v0.5 score: above + spillover_prevention_i

        The extra term rewards sectors that, if monitored, would
        reduce future risk propagation — not just current local uncertainty.

    Parameters
    ----------
    gamma          : softmax temperature (same role as in SoftmaxPolicy)
    outflow_weight : λ, weight of spillover prevention term
    learning_rate  : η, must match dynamics learning_rate for consistency
    """

    def __init__(
        self,
        gamma:          float = 0.5,
        outflow_weight: float = 0.5,
        learning_rate:  float = 0.15,
        risk_weighted:  bool  = False,
    ):
        self.gamma          = gamma
        self.outflow_weight = outflow_weight
        self.learning_rate  = learning_rate
        self.risk_weighted  = risk_weighted
        # risk_weighted=False → v0.5: path3 = Σⱼ wᵢⱼ
        # risk_weighted=True  → v0.6: path3 = Σⱼ wᵢⱼ·rⱼ  (downstream risk)

    def allocate(self, state: SystemState) -> Dict[str, float]:
        eta = self.learning_rate
        lam = self.outflow_weight

        scores = {}
        for sid, s in state.sectors.items():
            r  = s.risk
            u  = s.uncertainty
            c  = s.confidence
            a  = s.allocation

            if self.risk_weighted:
                # v0.6: weight each downstream edge by target's current risk
                # Σⱼ wᵢⱼ·rⱼ — not all spillover equally dangerous
                downstream = sum(
                    w * state.sectors[tgt].risk
                    for (src, tgt), w in state.coupling.items()
                    if src == sid
                )
            else:
                # v0.5: unweighted total outgoing coupling
                downstream = sum(
                    w for (src, tgt), w in state.coupling.items()
                    if src == sid
                )

            # Path 1: local r·u signal
            path1 = r * u

            # Path 2: confidence effect
            path2 = eta * (1 - c) * a * r

            # Path 3: spillover prevention (v0.5) or risk-weighted (v0.6)
            path3 = 2 * eta * lam * r * u * (1 - c) * downstream

            scores[sid] = path1 + path2 + path3

        return self._softmax(scores, self.gamma)


    def __repr__(self) -> str:
        version = "v0.6" if self.risk_weighted else "v0.5"
        return (f"PredictivePolicy({version}, γ={self.gamma}, "
                f"λ={self.outflow_weight}, η={self.learning_rate})")


class MPCPolicy(AdaptivePolicy):
    """
    ASAS v0.7 — Model Predictive Control allocation.

    Motivation
    ----------
    v0.5 and v0.6 use one-step analytical gradients to estimate
    the value of attending to each sector. Both fail because:

        1. One-step horizon cannot see delayed spillover benefits
           (Messe→Hbf prevention takes 3-4 steps to materialize)
        2. Risk ceiling clipping (r clipped to [0,1]) suppresses
           the gradient signal at high-risk sectors

    MPC solves both problems by directly simulating T steps forward
    under each candidate allocation, measuring the actual cumulative
    H(t) reduction — not an analytical approximation of it.

    Algorithm
    ---------
    For each sector i:
        1. Construct a "focused" allocation: more weight on sector i,
           less on others (controlled by focus_strength)
        2. Simulate T steps forward under this allocation
        3. Compute discounted cumulative H: V_i = Σₜ βᵗ · H(sₜ)

    Then allocate via softmax over {-V_i}:
        a_i* ∝ exp(-V_i / γ)   (lower V = better = higher allocation)

    This is a greedy one-step MPC: we pick the best current allocation
    assuming it will be held for T steps. In practice the policy is
    re-evaluated at every step, so the "held allocation" assumption
    only needs to be approximately true.

    Why this works where v0.5 failed
    ---------------------------------
    At horizon T=3: the simulation actually propagates spillover through
    the network. Attending to Messe at step 0 → c_Messe rises → u_Messe
    falls → w·r·u² spillover to Hbf decreases at step 1 → Hbf risk
    accumulates more slowly → H(t) at steps 2,3 is lower.
    The cumulative V_i captures this full causal chain. The analytical
    gradient in v0.5 could only see one link of it.

    Complexity
    ----------
    O(N × T) dynamics steps per allocation call.
    For N=5 sectors and T=5 horizon: 25 steps per call.
    Acceptable for real-time use at typical urban sensing frequencies.

    Parameters
    ----------
    horizon        : T, number of steps to simulate forward
    gamma          : softmax temperature over V_i scores
    focus_strength : how concentrated the candidate allocation is
                     0.0 = uniform for all candidates (degenerates to Equal)
                     1.0 = all attention on one sector (pure greedy probe)
                     0.3-0.6 = recommended range
    discount       : β, future H discount factor (1.0 = no discounting)
    dynamics_params: forwarded to dynamics.step() — must match engine params
    """

    def __init__(
        self,
        horizon:        int   = 5,
        gamma:          float = 0.5,
        focus_strength: float = 0.5,
        discount:       float = 0.95,
        **dynamics_params,
    ):
        self.horizon        = horizon
        self.gamma          = gamma
        self.focus_strength = focus_strength
        self.discount       = discount
        self.dynamics_params = dynamics_params

    def _candidate_allocation(
        self,
        state: SystemState,
        focus_sid: str,
    ) -> Dict[str, float]:
        """
        Build a candidate allocation that concentrates on focus_sid.

        Interpolates between uniform (focus_strength=0) and
        fully concentrated (focus_strength=1):

            a_focus = (1/N) + focus_strength · (1 - 1/N)
            a_other = (1/N) · (1 - focus_strength)
        """
        n       = state.N
        uniform = 1.0 / n
        alloc   = {}
        for sid in state.sector_ids:
            if sid == focus_sid:
                alloc[sid] = uniform + self.focus_strength * (1.0 - uniform)
            else:
                alloc[sid] = uniform * (1.0 - self.focus_strength)
        return alloc

    def _rollout(
        self,
        state: SystemState,
        a0: Dict[str, float],
        baseline: "AllocationPolicy",
    ) -> float:
        """
        Closed-loop rollout: simulate T steps forward, returning
        discounted cumulative H.

        Rollout structure (standard one-step MPC approximation):
            t=0: apply candidate allocation a0  ← the decision variable
            t=1..T-1: apply baseline_policy.allocate(sₜ)  ← receding horizon

        This is strictly more correct than open-loop (holding a0 fixed
        for all T steps), because:
            - Only a0 is the actual decision being evaluated
            - Future states will have their own policy re-evaluation
            - Open-loop overestimates the value of a0 by assuming it
              remains optimal throughout the horizon

        The baseline policy defines the "background" behaviour we assume
        the system will follow after the current step. Using EqualPolicy
        as baseline is conservative (assumes no future intelligence).
        Using SoftmaxPolicy as baseline is more optimistic.
        """
        from .dynamics import step as dynamics_step
        from .objective import system_entropy

        current      = state
        cumulative_H = 0.0
        beta         = 1.0

        for t in range(self.horizon):
            # Step 0: use candidate allocation (the decision variable)
            # Steps 1..T-1: use baseline policy (receding horizon)
            alloc = a0 if t == 0 else baseline.allocate(current)

            current = dynamics_step(
                state=current,
                allocation=alloc,
                external_signals=None,
                **self.dynamics_params,
            )
            cumulative_H += beta * system_entropy(current)
            beta *= self.discount

        return cumulative_H

    def allocate(self, state: SystemState) -> Dict[str, float]:
        """
        For each sector i, evaluate V_i = rollout value when a0 concentrates
        on sector i. Allocate via softmax over {-V_i}.

        The baseline policy used for steps 1..T is EqualPolicy (conservative
        assumption: after this step, attention returns to uniform).
        This makes V_i a lower bound on the true value of attending to i —
        a safe approximation for a risk-minimizing system.
        """
        baseline = EqualPolicy()
        values   = {}

        for sid in state.sector_ids:
            a0 = self._candidate_allocation(state, sid)
            values[sid] = -self._rollout(state, a0, baseline)
            # Negative: lower cumulative H → higher allocation priority

        return self._softmax(values, self.gamma)

    def __repr__(self) -> str:
        return (f"MPCPolicy(T={self.horizon}, γ={self.gamma}, "
                f"focus={self.focus_strength}, β={self.discount})")
