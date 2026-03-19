# The Evolution of the ASAS Framework (v0.1–v0.7)

The ASAS framework emerged through a sequence of conceptual and algorithmic
iterations, each motivated by empirical observation or theoretical limitation.
Rather than a single design step, the system evolved through a structured
research process involving model construction, boundary analysis, negative
results, and eventual algorithmic refinement.

## v0.1 — Problem Formulation

The starting point of ASAS was a practical observation: modern urban systems
generate abundant data across multiple sectors—transport, logistics,
infrastructure, and public safety—but no unified mechanism exists for
allocating monitoring attention across these domains under resource constraints.

The core problem was therefore framed as a **dynamic attention allocation
problem**:

* A monitoring system observes *N* sectors.
* Each sector produces risk signals.
* Monitoring resources are limited and cannot cover all sectors equally.
* The objective is to allocate attention to the sectors where it is most
  valuable at any given moment.

At this stage, the model contained only a single state variable per sector,
the systemic risk signal (r_i). No explicit representation of epistemic
uncertainty or system dynamics had yet been introduced.

---

## v0.2 — Separation of Systemic and Epistemic State

The first major conceptual breakthrough was the recognition that **risk and
knowledge about risk are fundamentally different quantities**.

To capture this distinction, the state representation was extended to two
layers:

* **Systemic risk** (r_i): the level of real-world disturbance in sector *i*.
* **Epistemic confidence** (c_i): how well the monitoring system understands
  the situation in sector *i*.

The epistemic uncertainty is defined as u_i = 1 - c_i.

This separation enables situations where:

* a sector is **high risk but well understood**, or
* a sector is **low risk but poorly understood**.

These cases require fundamentally different responses. This dual-layer state
representation became the conceptual foundation of the entire ASAS framework.

---

## v0.3 — Objective Functional and System Dynamics

With the two-layer state defined, the next step was to formalize the control
objective and system evolution.

### Objective Functional

The global objective was defined as **risk-weighted epistemic uncertainty**:

    H(t) = Σᵢ aᵢ(t) · rᵢ(t) · uᵢ(t)

The functional H(t) measures the residual epistemic risk exposure under
the current monitoring allocation.

### System Dynamics

Risk and confidence evolve over time according to coupled dynamics:

    rᵢ(t+1) = rᵢ(t)·(1−δ) + Σⱼ wᵢⱼ·rⱼ(t)·uⱼ(t)² − μ·aᵢ(t) + eᵢ(t)

    cᵢ(t+1) = cᵢ(t) + η·aᵢ(t)·(1−cᵢ(t)) − ρ·cᵢ(t)

Two structural modeling decisions were critical:

1. **Uncertain risk spreads faster than known risk**
   Spillover depends on rⱼuⱼ², meaning that high-risk sectors with
   low confidence are the most dangerous propagation sources.

2. **Confidence decay is necessary**
   The forgetting term ρ·cᵢ prevents confidence from saturating
   globally, which would otherwise eliminate epistemic differentiation
   between sectors.

---

## v0.4 — Reactive Softmax Allocation and the Symmetry Boundary

The first operational allocation strategy was implemented as a reactive
softmax policy:

    scoreᵢ = rᵢ·uᵢ + λ·outflowᵢ
    aᵢ* ∝ exp(scoreᵢ / γ)

where the outflow term identifies sectors that act as sources of network
spillover.

Experiments with this policy led to an important theoretical observation,
formalized as the **Symmetry Invariance Proposition**:

> Under symmetric network structure, symmetric initial conditions, and
> allocation policies that respect permutation symmetry, any reactive
> attention allocation policy produces uniform allocation.

This result establishes a **fundamental boundary**: intelligent reactive
allocation can only outperform uniform monitoring when **symmetry is broken**
by external signals such as incidents, strikes, or disasters.

---

## v0.5 — One-Step Predictive Allocation (Negative Result)

Motivated by the symmetry boundary of reactive policies, the next attempt
introduced a **one-step predictive allocation strategy**.

The policy estimated the marginal reduction of H via three causal paths:

1. Direct reduction of rᵢ·uᵢ
2. Confidence improvement through monitoring
3. Prevention of downstream spillover

However, experiments revealed that the predictive policy **performed worse
than uniform allocation**.

Two distortions were observed:

* **Confidence asymmetry distortion**
  Sectors with low confidence dominated the allocation score even when their
  systemic risk was modest.

* **Risk saturation masking**
  Clipping of risk values suppressed gradient signals in high-risk sectors.

More fundamentally, the results revealed that the benefits of spillover
prevention unfold across multiple time steps. A one-step predictive horizon
cannot capture these delayed effects. Both distortions are manifestations of
a single root cause: **one-step horizon myopia**.

---

## v0.6 — Risk-Weighted Predictive Correction (Failed Patch)

A subsequent modification attempted to correct the predictive model by
weighting spillover prevention by downstream sector risk.

Despite this adjustment, experimental performance deteriorated further.

The failure revealed that the fundamental limitation was not the weighting
structure but the **short predictive horizon** itself. Spillover prevention
generates benefits that accumulate over multiple future steps, which cannot
be captured by one-step gradient approximations.

---

## v0.7 — Model Predictive Attention Allocation

The breakthrough came with the adoption of a **Model Predictive Control
(MPC) formulation**.

Instead of estimating gradients analytically, the policy evaluates candidate
allocations by **simulating the system forward for multiple steps**.

For each sector i:

1. Construct a candidate allocation emphasizing sector i
2. Simulate system evolution for horizon T (closed-loop: step 0 uses
   candidate allocation, steps 1..T use baseline policy)
3. Compute cumulative discounted uncertainty:

       Vᵢ = Σₜ₌₁ᵀ βᵗ · H(sₜ)

The final allocation is computed as:

    aᵢ* ∝ exp(−Vᵢ / γ)

This multi-step evaluation captures delayed spillover prevention effects
that are invisible to one-step predictive policies.

### Horizon Sensitivity

Horizon sensitivity experiments were conducted for T ∈ {1,...,20}.
Results show that performance improvement relative to uniform allocation is
**empirically monotonic in the prediction horizon** across all tested
environments.

No finite optimal horizon was observed within the tested range. A practical
operating point was identified at **T = 5**: **15.7% improvement over
uniform allocation** at **5.6ms per decision step**.

Multi-step predictive reasoning is necessary to overcome the reactive
symmetry boundary identified in v0.4.

---

## Summary

| Version | Contribution |
|---------|-------------|
| v0.1 | Problem formulation: attention allocation under limited monitoring resources |
| v0.2 | Dual-layer state: systemic risk rᵢ + epistemic confidence cᵢ |
| v0.3 | Objective functional H(t) and coupled dynamics |
| v0.4 | Reactive softmax policy + symmetry invariance boundary theorem |
| v0.5 | One-step predictive allocation — negative result |
| v0.6 | Predictive weighting correction — failed patch |
| v0.7 | Model Predictive Control — multi-step breakthrough |

Negative results in v0.5 and v0.6 played a critical role in identifying
the structural limitation of short-horizon policies, directly motivating
the MPC formulation in v0.7.

The pattern: **hypothesis → experimental refutation → diagnosis → new theory**.
