from typing import Dict
import numpy as np


def normalize(scores: Dict[str, float]) -> Dict[str, float]:
    total = sum(scores.values())
    if total == 0:
        n = len(scores)
        return {k: 1.0 / n for k in scores}
    return {k: v / total for k, v in scores.items()}


# =========================
# ASAS Allocation
# =========================

def allocate_asas(
    risk: Dict[str, float],
    confidence: Dict[str, float],
    exploration_weight: float = 1.0,
    stabilization_weight: float = 0.5
) -> Dict[str, float]:

    raw_scores = {}

    for sector in risk:
        r = risk[sector]
        c = confidence[sector]

        exploration = exploration_weight * r * (1 - c)
        stabilization = stabilization_weight * r * c

        raw_scores[sector] = exploration + stabilization

    return normalize(raw_scores)


# =========================
# Baseline 1: Equal Allocation
# =========================

def allocate_equal(risk: Dict[str, float]) -> Dict[str, float]:
    n = len(risk)
    return {k: 1.0 / n for k in risk}


# =========================
# Baseline 2: Risk-Only
# =========================

def allocate_risk_only(risk: Dict[str, float]) -> Dict[str, float]:
    return normalize(risk)


# =========================
# Residual Entropy
# =========================

def calculate_residual_entropy(
    allocation: Dict[str, float],
    confidence: Dict[str, float]
) -> float:

    entropy = 0.0
    for sector in allocation:
        uncertainty = 1 - confidence[sector]
        entropy += allocation[sector] * uncertainty

    return entropy


# =========================
# Dynamic Confidence Update
# =========================

def update_confidence(
    confidence: Dict[str, float],
    allocation: Dict[str, float],
    learning_rate: float = 0.2
) -> Dict[str, float]:

    new_conf = {}

    for sector in confidence:
        c = confidence[sector]
        a = allocation[sector]
        new_c = c + learning_rate * a * (1 - c)
        new_conf[sector] = min(new_c, 1.0)

    return new_conf
