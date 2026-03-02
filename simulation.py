import numpy as np
from asas_core import (
    allocate_attention,
    calculate_residual_entropy,
    update_confidence
)

# Initial scenario

risk = {
    "Hbf": 0.95,
    "Bridges": 0.80,
    "Messe": 0.40,
    "Airport": 0.70
}

confidence = {
    "Hbf": 0.90,
    "Bridges": 0.60,
    "Messe": 0.30,
    "Airport": 0.50
}

history_entropy = []

print("=== ASAS Simulation Start ===")

for t in range(6):

    print(f"\n--- Iteration {t} ---")

    allocation = allocate_attention(
        risk,
        confidence,
        exploration_weight=1.0,
        stabilization_weight=0.5
    )

    uncertainty = {k: 1 - confidence[k] for k in confidence}

    entropy = calculate_residual_entropy(allocation, uncertainty)
    history_entropy.append(entropy)

    print("Allocation:")
    for k, v in allocation.items():
        print(f"  {k}: {v:.2f}")

    print(f"Residual Entropy: {entropy:.4f}")

    confidence = update_confidence(confidence, allocation)

print("\nEntropy Over Time:")
print(history_entropy)
