import copy
import matplotlib.pyplot as plt

from asas_core import (
    allocate_asas,
    allocate_equal,
    allocate_risk_only,
    compute_next_risk,
    update_confidence,
    calculate_entropy
)


def run_simulation(
    allocator,
    initial_risk,
    confidence,
    coupling,
    steps=40
):
    """
    Simulate a coupled urban risk system.

    Each iteration:
        1. Allocate resources
        2. Measure entropy
        3. Update risk (stateful dynamics)
        4. Update confidence
    """

    conf = copy.deepcopy(confidence)
    risk = copy.deepcopy(initial_risk)

    entropy_history = []

    for _ in range(steps):

        # 1️⃣ Allocation
        allocation = allocator(risk, conf)

        # 2️⃣ Measure system entropy
        entropy = calculate_entropy(risk, conf)
        entropy_history.append(entropy)

        # 3️⃣ Risk dynamics update
        risk = compute_next_risk(
            current_risk=risk,
            confidence=conf,
            allocation=allocation,
            coupling=coupling
        )

        # 4️⃣ Confidence learning
        conf = update_confidence(conf, allocation)

    return entropy_history


if __name__ == "__main__":

    initial_risk = {
        "Hbf": 0.9,
        "Bridges": 0.7,
        "Messe": 0.4,
        "Airport": 0.6
    }

    confidence = {
        "Hbf": 0.8,
        "Bridges": 0.5,
        "Messe": 0.2,
        "Airport": 0.4
    }

    # Strong systemic coupling
    coupling = {
        ("Messe", "Hbf"): 0.8,
        ("Hbf", "Bridges"): 0.7,
        ("Bridges", "Airport"): 0.6
    }

    asas_entropy = run_simulation(
        allocate_asas,
        initial_risk,
        confidence,
        coupling
    )

    equal_entropy = run_simulation(
        lambda r, c: allocate_equal(r),
        initial_risk,
        confidence,
        coupling
    )

    risk_entropy = run_simulation(
        lambda r, c: allocate_risk_only(r),
        initial_risk,
        confidence,
        coupling
    )

    plt.plot(asas_entropy)
    plt.plot(equal_entropy)
    plt.plot(risk_entropy)

    plt.legend(["ASAS", "Equal", "Risk-Only"])
    plt.title("Entropy Reduction in Stateful Coupled Urban Risk System")
    plt.xlabel("Iteration")
    plt.ylabel("System Entropy")
    plt.show()
