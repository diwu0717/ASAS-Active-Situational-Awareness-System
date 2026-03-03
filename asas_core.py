from typing import Dict, Tuple
import copy


# ============================================================
# Utility
# ============================================================

def normalize(scores: Dict[str, float]) -> Dict[str, float]:
    """
    Normalize dictionary values so they sum to 1.
    """
    total = sum(scores.values())
    if total == 0:
        n = len(scores)
        return {k: 1.0 / n for k in scores}
    return {k: v / total for k, v in scores.items()}


# ============================================================
# Allocation Strategies
# ============================================================

def allocate_asas(
    risk: Dict[str, float],
    confidence: Dict[str, float],
    exploration_weight: float = 1.0,
    stabilization_weight: float = 0.5
) -> Dict[str, float]:
    """
    ASAS allocation balances:
        - Exploration: focus on high risk + low confidence
        - Stabilization: maintain high risk + high confidence

    This avoids purely greedy risk chasing.
    """
    raw_scores = {}

    for s in risk:
        r = risk[s]
        c = confidence[s]

        exploration = exploration_weight * r * (1 - c)
        stabilization = stabilization_weight * r * c

        raw_scores[s] = exploration + stabilization

    return normalize(raw_scores)


def allocate_equal(risk: Dict[str, float]) -> Dict[str, float]:
    """
    Uniform resource allocation baseline.
    """
    n = len(risk)
    return {k: 1.0 / n for k in risk}


def allocate_risk_only(risk: Dict[str, float]) -> Dict[str, float]:
    """
    Greedy baseline:
    Allocate purely proportional to current risk.
    """
    return normalize(risk)


# ============================================================
# Stateful Coupled Risk Dynamics
# ============================================================

def compute_next_risk(
    current_risk: Dict[str, float],
    confidence: Dict[str, float],
    allocation: Dict[str, float],
    coupling: Dict[Tuple[str, str], float],
    natural_decay: float = 0.05,
    mitigation_strength: float = 0.4
) -> Dict[str, float]:
    """
    Dynamic risk update (STATEFUL MODEL):

    risk[t+1] =
        risk[t] * (1 - natural_decay)
      + spillover_from_low_confidence
      - mitigation_from_allocation

    This introduces:
        - Memory (risk accumulation)
        - Systemic propagation
        - Control feedback
    """

    next_risk = copy.deepcopy(current_risk)

    # 1️⃣ Natural decay (prevents unbounded growth)
    for s in next_risk:
        next_risk[s] *= (1 - natural_decay)

    # 2️⃣ Systemic spillover (nonlinear propagation)
    for (src, tgt), weight in coupling.items():
        spill = weight * (1 - confidence[src])**2
        next_risk[tgt] += spill

    # 3️⃣ Mitigation via allocation
    for s in next_risk:
        next_risk[s] -= mitigation_strength * allocation[s]
        next_risk[s] = max(next_risk[s], 0.0)

    return next_risk


# ============================================================
# Confidence Learning
# ============================================================

def update_confidence(
    confidence: Dict[str, float],
    allocation: Dict[str, float],
    learning_rate: float = 0.2
) -> Dict[str, float]:
    """
    Confidence increases when a sector receives monitoring resources.
    """

    new_conf = {}

    for s in confidence:
        c = confidence[s]
        a = allocation[s]

        new_c = c + learning_rate * a * (1 - c)
        new_conf[s] = min(new_c, 1.0)

    return new_conf


# ============================================================
# System Entropy
# ============================================================

def calculate_entropy(
    risk: Dict[str, float],
    confidence: Dict[str, float]
) -> float:
    """
    System-level uncertainty defined as:

        entropy = Σ (risk × uncertainty)

    where uncertainty = (1 - confidence)

    This ensures that:
        - High risk + low knowledge = high system cost
    """

    entropy = 0.0

    for s in risk:
        entropy += risk[s] * (1 - confidence[s])

    return entropy
