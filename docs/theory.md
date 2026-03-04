# ASAS — Theoretical Foundations

This document contains the mathematical basis of the ASAS framework:
state space definition, dynamics, objective functional, policy derivation,
fixed-point analysis, and the theoretical boundary result.

---

## 1. Notation

| Symbol | Type | Meaning |
|--------|------|---------|
| `N` | integer | Number of sectors |
| `i, j` | index | Sector indices, `i,j ∈ {1,...,N}` |
| `t` | integer | Discrete time step |
| `rᵢ(t)` | `[0,1]` | Systemic risk of sector i at time t |
| `cᵢ(t)` | `[0,1]` | Epistemic confidence of sector i at time t |
| `uᵢ(t)` | `[0,1]` | Uncertainty: `uᵢ = 1 − cᵢ` |
| `aᵢ(t)` | `[0,1]` | Attention allocation to sector i at time t |
| `wᵢⱼ` | `≥ 0` | Spillover coupling weight from sector i to j |
| `δ` | `(0,1)` | Natural risk decay rate |
| `μ` | `> 0` | Mitigation strength per unit attention |
| `η` | `(0,1)` | Confidence learning rate |
| `ρ` | `(0,1)` | Confidence forgetting rate |
| `λ` | `≥ 0` | Outflow weight in allocation score |
| `γ` | `> 0` | Softmax temperature (allocation regularization) |
| `eᵢ(t)` | `ℝ` | External event signal injected at sector i, time t |

Constraints: `Σᵢ aᵢ(t) = 1`, `aᵢ(t) ≥ 0` for all i, t.

---

## 2. State Space

ASAS represents each urban sector with three variables capturing two distinct layers:

**Systemic layer** — `rᵢ(t)`: what is happening in the world.
Driven by natural decay, spillover from neighboring sectors, mitigation from attention, and external event injection.

**Epistemic layer** — `cᵢ(t)`: what the system knows about what is happening.
Driven by learning from directed attention and structural forgetting (information staleness).

**Control variable** — `aᵢ(t)`: where sensing attention is allocated.
This is the *only* variable directly controlled by the system.

The separation of systemic and epistemic layers is the foundational design choice. It encodes the core claim:

> Urban failures often arise not from high systemic risk alone, but from the gap between systemic risk and epistemic confidence — the *epistemic gap* `uᵢ = 1 − cᵢ`.

---

## 3. Objective Functional

### 3.1 Definition

```
H(t) = Σᵢ  aᵢ(t) · rᵢ(t) · uᵢ(t)
```

`H(t)` is called **system entropy** in the ASAS codebase. This terminology is
intentionally metaphorical and must be carefully distinguished:

> **H(t) is not Shannon entropy.**
> It is a *risk-weighted uncertainty functional* — a weighted sum of epistemic
> gaps, where weights are the product of allocation and systemic risk.

Formally, `H(t)` belongs to the class of *control-weighted Lyapunov functionals*
used in regulation theory. Shannon entropy `−Σ p log p` measures distributional
uncertainty over outcomes; `H(t)` measures the residual epistemic gap that
remains after attention has been allocated.

The name "entropy" is retained because:
1. `H(t)` has the same qualitative behavior as thermodynamic entropy in a
   closed system — it tends to increase without active intervention.
2. Minimizing `H(t)` has the same intuitive structure as reducing disorder.
3. The notation distinguishes it from Shannon allocation entropy `H(a)` (Section 3.3).

### 3.2 Interpretation

`H(t)` measures the **marginal value of attention**: how much risk-weighted
uncertainty remains given current allocation.

- **H rising**: attention is misallocated; risk accumulates faster than awareness is built.
- **H stable**: the system maintains situational awareness under constant pressure.
- **H falling**: confidence is being built where risk is highest.

The system design objective is to **minimize cumulative H**:

```
min_{a(0),...,a(T)}  Σₜ H(t)
```

subject to `Σᵢ aᵢ(t) = 1`, `aᵢ(t) ≥ 0` for all t.

### 3.3 Allocation entropy (distinct from H(t))

```
H(a) = −Σᵢ  aᵢ · log(aᵢ)
```

This *is* Shannon entropy, applied to the allocation distribution.

- `H(a) = log(N)`: allocation is uniform (Equal policy)
- `H(a) = 0`: all attention concentrated on one sector

`H(a)` is used as a **concentration diagnostic**, not as part of the primary
objective. The ratio `H(a) / log(N)` is called the *Attention Utilization Index*
and measures how spread out attention is relative to the uniform baseline.

---

## 4. Dynamics

### 4.1 Risk evolution

```
rᵢ(t+1) = rᵢ(t)·(1−δ)
         + Σⱼ→ᵢ  wⱼᵢ · rⱼ(t) · uⱼ(t)²
         − μ · aᵢ(t)
         + eᵢ(t)
```

clipped to `[0, 1]`.

**Term 1 — natural decay** `rᵢ·(1−δ)`:
Risk dissipates at rate δ without intervention (e.g. crowd disperses, congestion clears).

**Term 2 — spillover** `Σⱼ→ᵢ wⱼᵢ · rⱼ(t) · uⱼ(t)²`:
Upstream sectors propagate risk proportional to their own systemic risk *and* epistemic
uncertainty. Both factors are necessary:
- `rⱼ`: a low-risk sector generates little spillover even if unmonitored
- `uⱼ²`: an unmonitored sector propagates more unpredictably (quadratic amplification)

**Term 3 — mitigation** `−μ·aᵢ`:
Directed attention reduces risk at rate μ per unit allocation.

**Term 4 — external injection** `eᵢ(t)`:
Exogenous risk signals from the perception layer (sensor anomalies, event
declarations, weather alerts). This term is what allows ASAS to respond to
real-world events, not only to internal dynamics.

**Explicit Euler stepping**:
The spillover term uses `rⱼ(t)` and `uⱼ(t)` — values from the *current* step,
before any updates are applied. This is forward Euler integration.

Consequence: a sector's rising risk does not immediately amplify its own
spillover within the same time step. This prevents implicit feedback
amplification, ensures numerical stability, and makes the dynamics
analytically tractable for the fixed-point analysis in Section 6.

### 4.2 Confidence evolution

```
cᵢ(t+1) = cᵢ(t)
         + η · aᵢ(t) · (1 − cᵢ(t))
         − ρ · cᵢ(t)
```

clipped to `[0, 1]`.

**Bounded learning** `η·aᵢ·(1−cᵢ)`:
Confidence grows at rate proportional to allocation, with saturation at cᵢ = 1
(standard bounded-growth form). The `(1−cᵢ)` factor ensures confidence cannot
exceed 1 and that returns to additional attention are diminishing as cᵢ rises.

**Structural forgetting** `−ρ·cᵢ`:
Confidence decays at rate ρ regardless of allocation. This models information
staleness: a sector not actively monitored becomes less well understood over time.

Forgetting is *structurally necessary*. Without it (`ρ = 0`), all `cᵢ → 1` as
`t → ∞`, `uᵢ → 0`, and the allocation score degenerates — the system loses the
ability to differentiate sectors by epistemic gap.

**Fixed point** of the confidence equation:
Setting `cᵢ(t+1) = cᵢ(t) = cᵢ*` and solving:

```
cᵢ* = η·aᵢ* / (ρ + η·aᵢ*)   ∈ (0, 1)  for all aᵢ* > 0
```

This fixed point always exists and is unique for any `aᵢ* > 0`. Its existence
is used in the boundary result derivation (Section 7).

---

## 5. Policy Space

### 5.1 Allocation score

The base priority signal for sector i is:

```
scoreᵢ = rᵢ · uᵢ  +  λ · outflowᵢ
```

where:

```
outflowᵢ = Σⱼ  wᵢⱼ · rᵢ · uᵢ²
```

**Score term 1** `rᵢ·uᵢ`: local AND signal. Both risk and uncertainty must be
high simultaneously. Multiplicative structure: a well-monitored sector (low uᵢ)
or a low-risk sector (low rᵢ) scores low even if the other factor is large.

**Score term 2** `λ·outflowᵢ`: systemic signal. Outflow measures how much
network-wide spillover sector i is currently generating. Attending to a
high-outflow sector prevents downstream risk accumulation. This term shifts
priority from risk sinks (which accumulate risk passively) to risk sources
(which generate it actively).

Note: `outflowᵢ` uses the same `r·u²` formula as the spillover term in
dynamics, evaluated at the current state. Policy reasoning and actual dynamics
are therefore grounded in the same physical quantity.

### 5.2 Entropy-regularized softmax (v0.4)

**Optimization problem**:

```
max_a  Σᵢ aᵢ · scoreᵢ  +  γ · H(a)
s.t.   Σᵢ aᵢ = 1,  aᵢ ≥ 0
```

The first term maximizes total allocated score — directing more attention toward
sectors with higher priority (higher `scoreᵢ`).
The second term maximizes allocation entropy `H(a) = −Σ aᵢ log aᵢ`, penalizing
excessive concentration.
`γ ≥ 0` is the regularization strength (temperature): higher γ increases the
penalty on concentration and spreads allocation toward uniform.

**Lagrangian** with multiplier `ν` for the simplex constraint:

```
L = Σᵢ aᵢ·scoreᵢ − γ·Σᵢ aᵢ·log(aᵢ) − ν·(Σᵢ aᵢ − 1)
```

**First-order condition** `∂L/∂aᵢ = 0`:

```
scoreᵢ − γ·(log(aᵢ) + 1) − ν = 0
log(aᵢ) = scoreᵢ/γ − 1 − ν/γ
aᵢ ∝ exp(scoreᵢ / γ)
```

This is the softmax function with temperature γ — no sign correction needed.
The derivation is internally consistent: higher score → higher allocation, as intended.

**Boundary behavior**:
- `γ → 0`: `aᵢ* → δ(argmax score)` — all attention on the highest-scoring sector (greedy)
- `γ → ∞`: `aᵢ* → 1/N` for all i — uniform allocation (Equal policy)

The Equal policy is therefore the `γ → ∞` limit of the softmax policy. It is
not an external baseline but a special case within the same parameter family.
This unification is a key structural result of the v0.4 formulation.

**No exploration floor needed**: softmax guarantees `aᵢ* > 0` for all i
(since `exp(x) > 0` for all x), replacing the heuristic ε-floor of earlier versions
with a mathematically grounded minimum-attention guarantee.

---

## 6. Fixed-Point Analysis

This section establishes the conditions under which the system reaches a
stationary distribution, as a prerequisite for the boundary result.

**Definition (stationary state)**: A state `(r*, c*, a*)` is stationary if
`rᵢ(t+1) = rᵢ(t) = rᵢ*`, `cᵢ(t+1) = cᵢ(t) = cᵢ*`, and `aᵢ` is fixed.

**Confidence fixed point** (from Section 4.2):

```
cᵢ* = η·aᵢ* / (ρ + η·aᵢ*)
uᵢ* = 1 − cᵢ* = ρ / (ρ + η·aᵢ*)
```

This is a monotone decreasing function of `aᵢ*`: more attention → higher
confidence → lower uncertainty. Uncertainty at the fixed point is bounded
below by 0 and above by `ρ/(ρ+η·min_aᵢ*)`.

**Risk fixed point**: Setting `rᵢ(t+1) = rᵢ(t) = rᵢ*` in the risk equation
(with `eᵢ = 0` for autonomous analysis):

```
0 = −δ·rᵢ*  +  Σⱼ→ᵢ wⱼᵢ·rⱼ*·(uⱼ*)²  −  μ·aᵢ*
```

Rearranging:

```
rᵢ* = [Σⱼ→ᵢ wⱼᵢ·rⱼ*·(uⱼ*)²  −  μ·aᵢ*] / δ
```

This is a linear system in `r*` given fixed `(a*, u*)`. Writing it in matrix form:

```
(δI − W̃) r* = μ a*
```

where `W̃` is the effective spillover matrix with entries `W̃ᵢⱼ = wⱼᵢ · (uⱼ*)²`.

A sufficient condition for a unique non-negative solution `r* ≥ 0` is that the
matrix `(δI − W̃)` is an M-matrix, which holds when:

```
ρ(W̃ / δ) < 1
```

where `ρ(·)` denotes the spectral radius. That is: **the spectral radius of the
effective spillover matrix must be strictly less than δ** (the natural decay rate).

This is the standard stability condition for linear positive systems and is
strictly stronger than the pointwise condition `δ > Σⱼ wⱼᵢ·(uⱼ*)²`, which is
only sufficient when the coupling matrix has a particular structure. The spectral
radius condition is the correct criterion for general network topologies.

---

## 7. Theoretical Boundary Result

### 7.1 Setup

**Definition (symmetric network)**: A network is *symmetric* if all sectors have
equal initial conditions and equal total coupling:

```
rᵢ(0) = r₀,  cᵢ(0) = c₀  for all i
Σⱼ wᵢⱼ = w  (equal total outflow weight)  for all i
Σⱼ wⱼᵢ = w  (equal total inflow weight)   for all i
```

**Definition (reactive policy)**: A policy is *reactive* if its allocation at
time t depends only on the state `(r(t), c(t))`, not on the history of states
or future predictions.

The softmax policy `aᵢ* ∝ exp(scoreᵢ/γ)` is reactive by this definition.

### 7.2 Proposition

**Proposition (Symmetry Invariance of Reactive Policies)**:

*Let the network be symmetric (Definition 7.1) and let the policy be the
softmax policy (Section 5.2) with any finite γ > 0. Then, for all t ≥ 0:*

```
aᵢ(t) = 1/N  for all i
```

*Equivalently: the only symmetry-consistent allocation trajectory under any
reactive softmax policy in a symmetric network is uniform allocation,
regardless of γ.*

**Remark on scope**: This is a *symmetry invariance* argument, not a global
optimality proof. The Proposition establishes that reactive policies cannot
break the symmetry of a symmetric network — uniform is the unique
symmetry-consistent solution. It does not directly prove that uniform minimizes
cumulative H(t) over all possible (including asymmetric) policies. The
practical consequence is identical: in symmetric environments, no reactive
policy can outperform Equal by construction, since they all produce the same
allocation trajectory.

**Proof sketch**:

By symmetry, the system initialized with equal conditions evolves identically
across all sectors at every step. Formally, if `rᵢ(t) = r(t)` and
`cᵢ(t) = c(t)` for all i at time t, then by the risk and confidence equations
(which are symmetric in i under the symmetric network condition):

```
rᵢ(t+1) = r(t+1)  and  cᵢ(t+1) = c(t+1)  for all i
```

By induction, `rᵢ(t) = r(t)` and `cᵢ(t) = c(t)` for all i, t.

Consequently, `scoreᵢ(t) = score(t)` for all i at every step.

The softmax allocation then gives:

```
aᵢ(t) = exp(score(t)/γ) / Σⱼ exp(score(t)/γ) = 1/N  for all i
```

This holds for all t, including the stationary state. □

### 7.3 Corollary

*Under a symmetric network, the cumulative entropy satisfies:*

```
Σₜ H(t)|_{softmax} = Σₜ H(t)|_{equal}
```

*for any finite γ > 0.*

In particular, no reactive softmax policy with finite γ outperforms the Equal
policy in a symmetric network. The γ-scan result (monotone improvement as
γ → ∞, with minimum at γ = ∞) is a direct consequence.

### 7.4 Assumptions and limitations

The Proposition depends on the following assumptions:

| Assumption | Status | Consequence if violated |
|---|---|---|
| Network symmetry | Imposed by definition | Breaks symmetry → uniform no longer optimal |
| No external events (`eᵢ = 0`) | Autonomous analysis | Events break symmetry → ASAS gains advantage |
| Reactive policy | Definition | Predictive policies may outperform uniform |
| Linear confidence dynamics | Current model | Nonlinear `uᵢ = f(cᵢ)` could change fixed point |

**Implication for ASAS deployment**:

The theoretical boundary result identifies the *operating condition* for ASAS
advantage over uniform monitoring:

> ASAS outperforms uniform allocation if and only if network asymmetry is present —
> either structurally (heterogeneous coupling) or dynamically (external event injection).

This is not a limitation of ASAS; it is a precise characterization of when
intelligent allocation adds value. Urban systems under normal operation are
approximately symmetric; under crisis conditions (strikes, events, weather
extremes), they become highly asymmetric. ASAS is designed for crisis conditions.

---

## 8. Version History and Failure Analysis

Each version of the allocation policy corresponds to a testable hypothesis about
what signal should drive attention allocation. The failure of each hypothesis
motivated the next version.

### v0.1 — Pure reactive: `score = r·u`

**Hypothesis**: Sectors with high risk AND high uncertainty warrant the most attention.

**Result**: Converges to Risk-Only attractor as confidence rises. Without forgetting,
`cᵢ → 1`, `uᵢ → 0`, and the score degenerates to `rᵢ`.

**Fix**: Introduce forgetting rate ρ > 0, preventing confidence saturation.

### v0.2 — Inflow-aware: `score = r·u + λ·inflow`

**Hypothesis**: Sectors under high spillover pressure deserve more attention.

**Experimental result**: Cumulative H(t) *increased* vs. Equal by 120–147%.

**Diagnosis**: `inflow_i` is highest at structural risk sinks (Airport receives
spillover from Messe + Bridges). The policy directed attention to the *symptom*
(accumulated risk at sink) rather than the *cause* (low-confidence source generating
spillover). This created a positive feedback loop:
Airport inflow highest → ASAS gives Airport more resources → upstream sectors
under-monitored → their `(1−c)²` stays high → spillover to Airport persists.

**Fix**: Replace inflow with outflow — the signal that identifies sources, not sinks.

### v0.3 — Source-aware: `score = r·u + λ·outflow − μ·centrality`

**Hypothesis**: Attention should go to sectors that, if monitored, would reduce
network-wide spillover.

**Experimental result**: 12–20% improvement over v0.2. Airport allocation dropped
from 55% to 46%. But still lost to Equal by 50–72%.

**Diagnosis**: Even with outflow signal, Airport's rising `r·u` continued to attract
attention via the score term. Centrality penalty was insufficient to fully counter this.
More fundamentally: the score still optimized current H(t), not future H(t+1).

### v0.4 — Entropy-regularized: `a* ∝ exp(score/γ)`

**Hypothesis**: Concentration is the structural problem. Penalize it directly in the
objective function, not through heuristic caps or floors.

**Experimental result**: γ-scan shows strict monotone improvement as γ increases,
with minimum at γ → ∞ (uniform). The theoretical boundary result (Section 7)
explains why: in a symmetric network, reactive policies cannot outperform uniform.

**Interpretation**: v0.4 did not fail — it *proved* the limit of reactive allocation.
The γ → ∞ convergence is the theoretically correct answer to the question
"what is the best reactive policy under symmetric conditions?"

### v0.5 — Predictive (planned)

**Hypothesis**: The reactive framework fails because it minimizes `H(t)` rather
than `H(t+1)`. A one-step predictive policy should use:

```
ΔHᵢ = rᵢ·uᵢ − Σⱼ wᵢⱼ · rᵢ · (∂uᵢ/∂aᵢ) · uᵢ
```

The second term captures the future spillover reduction from attending to sector i.
This is the "causal gradient" absent from all reactive policies.

Under the symmetric network condition, this gradient is also symmetric, so uniform
allocation remains optimal. Predictive policies gain advantage only under asymmetric
conditions — the same operating domain as ASAS.

---

## 9. Summary of Parameters

| Parameter | Symbol | Default | Effect |
|---|---|---|---|
| Natural decay | δ | 0.05 | Higher → risk dissipates faster |
| Mitigation strength | μ | 0.30 | Higher → attention reduces risk more |
| Learning rate | η | 0.15 | Higher → confidence builds faster |
| Forgetting rate | ρ | 0.03 | Higher → confidence decays faster; necessary for non-degenerate dynamics |
| Outflow weight | λ | 0.50 | Higher → policy prioritizes network sources over local risk |
| Temperature | γ | 0.50 | Higher → allocation spreads toward uniform; γ→∞ recovers Equal policy |

---

*For experimental results corresponding to this theory, see `benchmark/allocation_comparison.py`.*
*For the reference deployment scenario, see `examples/frankfurt_strike/`.*
