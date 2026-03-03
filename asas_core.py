from typing import Dict, Tuple
import copy


# ============================================================
# Utility
# ============================================================

def normalize(scores: Dict[str, float]) -> Dict[str, float]:
    total = sum(scores.values())
    if total == 0:
        n = len(scores)
        return {k: 1.0 / n for k in scores}
    return {k: v / total for k, v in scores.items()}


# ============================================================
# Allocation Strategies
# ============================================================

def allocate_asas(risk, confidence,
                  exploration_weight=1.0,
                  stabilization_weight=0.5):

    raw = {}

    for s in risk:
        r = risk[s]
        c = confidence[s]

        exploration = exploration_weight * r * (1 - c)
        stabilization = stabilization_weight * r * c

        raw[s] = exploration + stabilization

    return normalize(raw)


def allocate_equal(risk):
    n = len(risk)
    return {k: 1.0 / n for k in risk}


def allocate_risk_only(risk):
    return normalize(risk)


# ============================================================
# Coupled Risk World
# ============================================================

def compute_next_risk(
    base_risk: Dict[str, float],
    current_risk: Dict[str, float],
    confidence: Dict[str, float],
    allocation: Dict[str, float],
    coupling: Dict[Tuple[str, str], float],
    mitigation_strength: float = 0.4
):
    """
    Risk dynamics:

    next_risk =
        intrinsic_base
      + spillover_from_low_confidence
      - mitigation_from_allocation

    This creates a real control problem.
    """

    next_risk = copy.deepcopy(base_risk)

    # --- Spillover (systemic propagation) ---
    for (src, tgt), weight in coupling.items():
        spill = weight * (1 - confidence[src])**2
        next_risk[tgt] += spill

    # --- Mitigation (monitoring reduces risk) ---
    for s in next_risk:
        mitigation = mitigation_strength * allocation[s]
        next_risk[s] -= mitigation

        # Risk bounded below
        next_risk[s] = max(next_risk[s], 0.0)

    return next_risk


# ============================================================
# Confidence Learning
# ============================================================

def update_confidence(confidence, allocation, learning_rate=0.2):

    new_conf = {}

    for s in confidence:
        c = confidence[s]
        a = allocation[s]

        new_c = c + learning_rate * a * (1 - c)
        new_conf[s] = min(new_c, 1.0)

    return new_conf


# ============================================================
# True System Entropy
# ============================================================

def calculate_entropy(risk, confidence):
    """
    System entropy defined as:

        risk × uncertainty

    This ensures:
    - High risk + low knowledge = dangerous
    """

    entropy = 0.0

    for s in risk:
        entropy += risk[s] * (1 - confidence[s])

    return entropy
