import copy
import matplotlib.pyplot as plt

from asas_core import (
    allocate_asas,
    allocate_equal,
    allocate_risk_only,
    calculate_residual_entropy,
    update_confidence
)


def run_simulation(allocator, risk, confidence, steps=20):

    conf = copy.deepcopy(confidence)
    entropy_history = []

    for _ in range(steps):
        allocation = allocator(risk, conf)
        entropy = calculate_residual_entropy(allocation, conf)
        entropy_history.append(entropy)
        conf = update_confidence(conf, allocation)

    return entropy_history


if __name__ == "__main__":

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

    asas_entropy = run_simulation(allocate_asas, risk, confidence)
    equal_entropy = run_simulation(lambda r, c: allocate_equal(r), risk, confidence)
    risk_entropy = run_simulation(lambda r, c: allocate_risk_only(r), risk, confidence)

    plt.plot(asas_entropy)
    plt.plot(equal_entropy)
    plt.plot(risk_entropy)

    plt.legend(["ASAS", "Equal", "Risk-Only"])
    plt.title("Entropy Reduction Benchmark")
    plt.xlabel("Iteration")
    plt.ylabel("Residual Entropy")
    plt.show()
