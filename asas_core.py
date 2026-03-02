from typing import Dict
import numpy as np


def normalize(scores: Dict[str, float]) -> Dict[str, float]:
    total = sum(scores.values())
    if total == 0:
        n = len(scores)
        return {k: 1.0 / n for k in scores}
    return {k: v / total for k, v in scores.items()}


def allocate_attention(
    risk: Dict[str, float],
    confidence: Dict[str, float],
    exploration_weight: float = 1.0,
    stabilization_weight: float = 0.5,
    min_allocation: float = 0.0
) -> Dict[str, float]:
    """
    ASAS v0.1 Core Allocation Engine

    Attention Score_i =
        exploration_weight * Risk_i * (1 - Confidence_i)
        +
        stabilization_weight * Risk_i * Confidence_i

    exploration_weight controls priority of uncertainty reduction.
    stabilization_weight controls priority of confirmed risk control.
    """

    raw_scores = {}

    for sector in risk:
        r = risk[sector]
        c = confidence[sector]

        exploration = exploration_weight * r * (1 - c)
        stabilization = stabilization_weight * r * c

        raw_scores[sector] = exploration + stabilization

    allocations = normalize(raw_scores)

    if min_allocation > 0:
        allocations = {
            k: min_allocation + (1 - min_allocation) * v
            for k, v in allocations.items()
        }
        allocations = normalize(allocations)

    return allocations


def calculate_residual_entropy(
    allocation: Dict[str, float],
    uncertainty: Dict[str, float]
) -> float:
    """
    Residual Entropy:
    Sum_i ( Allocation_i × Uncertainty_i )

    Uncertainty_i typically = (1 - Confidence_i)
    """

    entropy = 0.0
    for sector in allocation:
        entropy += allocation[sector] * uncertainty[sector]

    return entropy


def update_confidence(
    confidence: Dict[str, float],
    allocation: Dict[str, float],
    learning_rate: float = 0.2
) -> Dict[str, float]:
    """
    Simple dynamic update:
    More attention → faster confidence improvement.
    """

    new_conf = {}

    for sector in confidence:
        c = confidence[sector]
        a = allocation[sector]

        new_c = c + learning_rate * a * (1 - c)
        new_conf[sector] = min(new_c, 1.0)

    return new_conf
