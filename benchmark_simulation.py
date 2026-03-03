import copy
import matplotlib.pyplot as plt

from asas_core import (
    allocate_asas,
    allocate_equal,
    allocate_risk_only,
    calculate_residual_entropy,
    update_confidence,
    compute_effective_risk
)


def run_simulation(allocator, base_risk, confidence, coupling, steps=20):
    """
    Run multi-step simulation of entropy reduction.

    Each iteration consists of:

        1. Compute effective risk (with cross-domain spillover)
        2. Allocate monitoring resources
        3. Measure residual entropy
        4. Update confidence

    This models a cognitive control loop.
    """

    conf = copy.deepcopy(confidence)
    entropy_history = []

    for _ in range(steps):

        # Step 1: Risk evolves via systemic coupling
        effective_risk = compute_effective_risk(base_risk, conf, coupling)

        # Step 2: Allocation strategy decides resource distribution
        allocation = allocator(effective_risk, conf)

        # Step 3: Measure residual entropy
        entropy = calculate_residual_entropy(allocation, conf)
        entropy_history.append(entropy)

        # Step 4: Update system knowledge state
        conf = update_confidence(conf, allocation)

    return entropy_history


if __name__ == "__main__":

    # Base intrinsic risk levels
    base_risk = {
        "Hbf": 0.95,
        "Bridges": 0.80,
        "Messe": 0.40,
        "Airport": 0.70
    }

    # Initial knowledge confidence
    confidence = {
        "Hbf": 0.90,
        "Bridges": 0.60,
        "Messe": 0.30,
        "Airport": 0.50
    }

    # Cross-domain systemic coupling
    # (source -> target)
    coupling = {
        ("Hbf", "Bridges"): 0.25,
        ("Bridges", "Airport"): 0.20,
        ("Messe", "Hbf"): 0.15
    }

    # Run strategies
    asas_entropy = run_simulation(
        allocate_asas,
        base_risk,
        confidence,
        coupling
    )

    equal_entropy = run_simulation(
        lambda r, c: allocate_equal(r),
        base_risk,
        confidence,
        coupling
    )

    risk_entropy = run_simulation(
        lambda r, c: allocate_risk_only(r),
        base_risk,
        confidence,
        coupling
    )

    # Visualization
    plt.plot(asas_entropy)
    plt.plot(equal_entropy)
    plt.plot(risk_entropy)

    plt.legend(["ASAS", "Equal", "Risk-Only"])
    plt.title("Entropy Reduction Benchmark (Coupled Risk World)")
    plt.xlabel("Iteration")
    plt.ylabel("Residual Entropy")
    plt.show()
