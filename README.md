# ASAS — Active Situational Awareness System

**A control-theoretic framework for dynamic attention allocation in coupled risk networks.**

> *ASAS makes the city legible to decision-makers.*

---

## What it does

Urban systems generate continuous risk signals across interdependent sectors.
The problem is not missing data — it is that no system is responsible for
integrating meaning across domains under pressure.

ASAS formalizes this as a control problem:

- Each sector has two state layers: **systemic risk** `r_i` ("what is happening")
  and **epistemic confidence** `c_i` ("how well we know it")
- **Attention allocation** `a_i` is the sole control variable (Σ aᵢ = 1)
- The objective is to minimize **risk-weighted uncertainty**:

```
H(t) = Σᵢ  aᵢ(t) · rᵢ(t) · uᵢ(t)       where uᵢ = 1 - cᵢ
```

The policy answers one question:
*Given limited monitoring resources, where should attention go right now?*

---

## Real-World Validation

Tested on the **Feb 2, 2026 Frankfurt Ver.di transport strike** —
a real, unplanned event. The system ingested thermal, acoustic, and flow
sensor anomalies, grounded them against live news sources, generated a
structured operational report, and triggered drone reconnaissance when
Global Entropy crossed 76%.

The system read LED signage at Frankfurt Hauptbahnhof in German
(*"F wird bestreikt!"*) and integrated it into its situational picture.

> This was not a scripted demo. The strike happened. ASAS responded.

**Origin**: This project grew out of a direct experience at the 2025 Frankfurt
Marathon, where a systemic coordination failure left an injured elite runner
without thermal protection for over 30 minutes — despite all relevant
information existing across weather, medical, and logistics systems.
No unified situational picture existed. ASAS is the formalization of
what was missing that day.

---

## Policy Evolution

The theoretical core of ASAS is a systematic policy development chain.
Each version is experimentally tested; failures are diagnosed and documented.

| Version | Policy | Result | Key Finding |
|---------|--------|--------|-------------|
| v0.4 | SoftmaxPolicy | ✓ Baseline | Symmetry invariance theorem: under symmetric networks, all reactive policies converge to uniform allocation |
| v0.5 | PredictivePolicy | ❌ Negative result | Confidence asymmetry distortion + risk saturation masking |
| **v0.7** | **MPCPolicy** | **✓ +14–47% over Equal** | Multi-step horizon captures delayed spillover benefits |

The v0.5 negative result is **theoretically informative** — it identifies
one-step myopia as the structural barrier and directly motivates v0.7.
See [`docs/theory.md`](docs/theory.md) Section 8 for full diagnosis.

---

## ASAS-MPC Policy (v0.7)

For each sector i, construct a counterfactual allocation,
simulate T steps forward under closed-loop rollout, and allocate
via softmax over negative cumulative H:

```
Candidate:   aᵢ_j = 1/N + f·(1-1/N)   if j = i
                   = 1/N·(1-f)          if j ≠ i

Rollout:     V_i = Σₜ₌₁ᵀ  βᵗ · H(sₜ)

Allocation:  aᵢ* ∝ exp(-V_i / γ)
```

**Horizon sensitivity**: H(T) is strictly monotonically decreasing in T.
No optimal finite horizon exists. T=5 is the efficiency frontier knee:
15.7% improvement at 5.6ms per step.

| T | Improvement vs Equal | Runtime |
|---|---|---|
| 1 | 14.4% | 1.2ms |
| 3 | 14.9% | 3.0ms |
| **5** | **15.7%** | **5.6ms** ← recommended |
| 10 | 19.2% | 8.7ms |
| 20 | 22.8% | 16.5ms |

**Complexity**: O(N²T) per step.
Full simplex search would require O(ε^{-(N-1)}) — approximately 10¹⁹
evaluations for N=20, ε=0.1.

---

## Quickstart

```bash
git clone https://github.com/diwu0717/ASAS-Active-Situational-Awareness-System
cd ASAS-Active-Situational-Awareness-System
pip install -r requirements.txt

# Run Frankfurt strike scenario
python examples/frankfurt_strike/scenario.py

# Run full policy benchmark
python benchmark/allocation_comparison.py

# Run horizon sensitivity analysis (T=1 to 20)
python benchmark/horizon_sensitivity.py
```

---

## Project Structure

```
ASAS-Active-Situational-Awareness-System/
├── asas/
│   ├── core/
│   │   ├── state.py          # SystemState: sector risk + confidence
│   │   ├── dynamics.py       # Risk propagation: w_ij · r_j · u_j²
│   │   ├── objective.py      # H(t) = Σ aᵢrᵢuᵢ
│   │   ├── policy.py         # Equal → Softmax → Predictive → MPCPolicy
│   │   └── engine.py         # Simulation engine
│   └── cognitive/
│       ├── base.py           # CognitiveHub base class
│       └── claude.py         # LLM operational report adapter
├── benchmark/
│   ├── allocation_comparison.py    # Full policy comparison
│   ├── horizon_sensitivity.py      # T=1..20 sweep + efficiency frontier
│   ├── policy_comparison.png
│   └── horizon_sensitivity.png
├── examples/
│   └── frankfurt_strike/
│       └── scenario.py       # Feb 2, 2026 validation scenario
└── docs/
    └── theory.md             # Complete theoretical derivation (1000+ lines)
```

---

## Results

Policy comparison across four coupling environments
(lower cumulative H is better):

```
                  STABLE   MARGINAL   UNSTABLE   EXPLOSIVE
Equal (baseline)    2.63      3.99       5.00       6.17
Softmax v0.4        2.64      4.43       5.64       7.36
Predictive v0.5      —         —         5.99        —      ← worse than Equal
MPC T=5 (v0.7)      2.28      2.74       4.22        —      ✓ beats Equal
```

MPC advantage scales with coupling strength:
stronger networks create larger delayed spillover effects
that the multi-step horizon captures.

---

## Theory

Complete theoretical development in [`docs/theory.md`](docs/theory.md):

- **Section 6**: Stability conditions and spectral radius analysis
- **Section 7**: Symmetry Invariance Proposition (v0.4 boundary theorem)
- **Section 8**: v0.5 failure analysis — confidence asymmetry distortion,
  risk saturation masking, one-step horizon limitation
- **Section 8b**: ASAS-MPC formal definition, approximation error bounds,
  horizon monotonicity, computational complexity

Workshop paper draft: [`docs/abstract_v07.md`](docs/abstract_v07.md)

---

## Inspiration

This framework was motivated by a direct experience at the
**2025 Frankfurt Marathon**. An elite runner withdrew due to injury
in 6–10°C wind chill, sat without thermal protection, and waited
over 30 minutes for aid. Volunteers called for help but lacked equipment.
Ambulances were delayed by road closures.

All relevant information existed — weather data, road closure maps,
medical station locations. No system integrated it into a unified
situational picture.

ASAS is the formalization of what was missing that day.

---

## License

MIT

---

*For questions or collaboration: open an issue or reach out directly.*
