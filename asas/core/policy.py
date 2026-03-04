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
