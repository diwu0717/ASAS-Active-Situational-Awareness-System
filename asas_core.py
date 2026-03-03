from typing import Dict, Tuple
import numpy as np


# ============================================================
# Utility: Normalization
# ============================================================

def normalize(scores: Dict[str, float]) -> Dict[str, float]:
    """
    Normalize a dictionary of scores into a probability distribution.

    Why?
    ----
    Allocation is modeled as a resource distribution problem.
    The total available monitoring resource is fixed (sum = 1.0).

    If all scores are zero, we fall back to uniform allocation.
    """

    total = sum(scores.values())

    if total == 0:
        n = len(scores)
        return {k: 1.0 / n for k in scores}

    return {k: v / total for k, v in scores.items()}


# ============================================================
# ASAS Allocation Strategy
# ============================================================

def allocate_asas(
    risk: Dict[str, float],
    confidence: Dict[str, float],
    exploration_weight: float = 1.0,
    stabilization_weight: float = 0.5
) -> Dict[str, float]:
    """
    ASAS allocation strategy.

    Core idea:
    ----------
    Monitoring effort is distributed based on BOTH:

        1. Risk magnitude (r)
        2. Confidence level (c)

    Two components:

        Exploration   -> r * (1 - c)
            Focus on uncertain but risky sectors.
            Prevent hidden or emerging risks.

        Stabilization -> r * c
            Maintain stability in already monitored high-risk areas.

    The weighting allows balancing between proactive discovery
    and maintaining control of known threats.
    """

    raw_scores = {}

    for sector in risk:

        r = risk[sector]
        c = confidence[sector]

        # Explore sectors where risk is high but confidence is low
        exploration = exploration_weight * r * (1 - c)

        # Stabilize sectors that are high-risk but already well understood
        stabilization = stabilization_weight * r * c

        raw_scores[sector] = exploration + stabilization

    return normalize(raw_scores)


# ============================================================
# Baseline 1: Equal Allocation
# ============================================================

def allocate_equal(risk: Dict[str, float]) -> Dict[str, float]:
    """
    Uniform resource distribution.

    Serves as naive baseline:
    - Ignores risk
    - Ignores confidence
    """

    n = len(risk)
    return {k: 1.0 / n for k in risk}


# ============================================================
# Baseline 2: Risk-Only (Greedy)
# ============================================================

def allocate_risk_only(risk: Dict[str, float]) -> Dict[str, float]:
    """
    Greedy strategy.

    Allocates purely proportional to observed risk.

    Assumption:
    -----------
    Highest current risk deserves most attention.

    Limitation:
    -----------
    Ignores uncertainty structure and cross-domain effects.
    """

    return normalize(risk)


# ============================================================
# Residual Entropy
# ============================================================

def calculate_residual_entropy(
    allocation: Dict[str, float],
    confidence: Dict[str, float]
) -> float:
    """
    Residual system entropy.

    Interpretation:
    ---------------
    Entropy represents remaining system uncertainty.

    For each sector:
        uncertainty = (1 - confidence)

    The total entropy is allocation-weighted uncertainty.

    Why weighted?
    -------------
    Because entropy is measured relative to
    where the system is focusing its attention.

    If resources are allocated to highly uncertain areas,
    entropy is high.
    """

    entropy = 0.0

    for sector in allocation:
        uncertainty = 1 - confidence[sector]
        entropy += allocation[sector] * uncertainty

    return entropy


# ============================================================
# Confidence Update (Learning Model)
# ============================================================

def update_confidence(
    confidence: Dict[str, float],
    allocation: Dict[str, float],
    learning_rate: float = 0.2
) -> Dict[str, float]:
    """
    Confidence update rule.

    Model assumption:
    -----------------
    Monitoring effort increases confidence.

    Learning dynamic:
        new_c = c + alpha * allocation * (1 - c)

    Properties:
    -----------
    - Diminishing returns (bounded by 1.0)
    - No overshoot
    - Stability guaranteed
    """

    new_conf = {}

    for sector in confidence:

        c = confidence[sector]
        a = allocation[sector]

        new_c = c + learning_rate * a * (1 - c)

        new_conf[sector] = min(new_c, 1.0)

    return new_conf


# ============================================================
# Coupled Risk Dynamics (Systemic Risk Model)
# ============================================================

def compute_effective_risk(
    base_risk: Dict[str, float],
    confidence: Dict[str, float],
    coupling: Dict[Tuple[str, str], float]
) -> Dict[str, float]:
    """
    Compute effective risk under cross-domain coupling.

    Core idea:
    ----------
    Urban risks are not independent.

    If sector A has low confidence,
    it can propagate risk into sector B.

    coupling[(A, B)] = weight
        means A influences B with given intensity.

    Spillover model:
        spillover = weight * (1 - confidence[A])

    This creates SYSTEMIC risk,
    not visible in isolated sector models.
    """

    effective = base_risk.copy()

    for (src, tgt), weight in coupling.items():

        spillover = weight * (1 - confidence[src])

        effective[tgt] += spillover

    return effective
