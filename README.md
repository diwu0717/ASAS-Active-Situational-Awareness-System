# ASAS — Active Situational Awareness System

> *"Many urban risks remain invisible not because data is missing,  
> but because no system is responsible for integrating meaning across domains."*

---

## Problem

Existing urban monitoring systems treat risk signals **independently and reactively**. They observe what is happening, but do not model:

- **Spillover effects** — how low-confidence sectors propagate uncertainty to neighbors
- **Epistemic uncertainty as a controllable variable** — confidence is not fixed; it responds to where attention is directed
- **Attention allocation as a dynamic control problem** — where to look next is a decision with system-wide consequences

This leads to two failure modes observed in practice:

| Failure mode | Cause | Consequence |
|---|---|---|
| **Uniform monitoring** | No priority signal | High-risk sectors under-observed |
| **Risk-sink concentration** | Reactive focus on visible hotspots | Upstream causes ignored; sinks accumulate risk regardless |

The Frankfurt Marathon incident illustrates the second failure: a professional athlete collapsed at an aid station while all relevant data — weather, road closures, event logistics — existed in separate systems. No mechanism integrated these signals into a unified risk picture.

**ASAS formulates situational awareness as a control-theoretic optimization problem**, where the system actively manages the allocation of sensing attention to minimize risk-weighted uncertainty across an urban sector network.

---

## Contributions

ASAS introduces:

1. **A unified state representation** separating *systemic risk* (what is happening) from *epistemic confidence* (what the system knows), making the epistemic gap an explicit, controllable quantity.

2. **A control-theoretic formulation of attention allocation** — where to direct sensing resources is treated as a dynamic control variable, not a static dashboard parameter.

3. **An entropy-based objective function** `H(t) = Σ aᵢ rᵢ uᵢ` linking allocation decisions directly to residual uncertainty under risk.

4. **A modular policy space** that continuously interpolates between uniform and greedy allocation via a single temperature parameter γ, with theoretical guarantees on boundary behavior.

5. **A theoretical boundary result**: under persistent symmetric spillover with linear confidence dynamics, reactive entropy minimization converges to uniform allocation. ASAS advantage over uniform emerges precisely when external event signals break network symmetry.

6. **A plug-in cognitive reasoning layer** — any LLM can be attached as an optional adapter to translate quantitative state into human-readable situational intelligence, without coupling the mathematical core to any specific model.

---

## Architecture

```
ASAS/
├── asas/
│   ├── core/                  ← Mathematical framework (the theoretical contribution)
│   │   ├── state.py           ← SystemState: systemic risk + epistemic confidence
│   │   ├── dynamics.py        ← State transition equations (explicit Euler)
│   │   ├── objective.py       ← H(t): entropy objective + diagnostics
│   │   ├── policy.py          ← Pluggable allocation strategies
│   │   └── engine.py          ← Control loop orchestrator (no math inside)
│   └── cognitive/             ← LLM adapter layer (optional, thin)
│       ├── base.py            ← CognitiveHub abstract interface (one method)
│       ├── claude.py          ← Claude (Anthropic) reference implementation
│       └── gemini.py          ← Gemini (Google) reference implementation
├── examples/
│   └── frankfurt_strike/      ← Reference scenario: Feb 2, 2026 transport strike
├── benchmark/
│   └── allocation_comparison.py  ← Reproduces policy evolution experiments
└── docs/
    └── theory.md              ← Full mathematical derivations
```

**Design principle**: `core/` is the theoretical contribution. `cognitive/` is a thin adapter. The LLM is not the system — it is one possible reasoning backend. Replacing or removing the cognitive layer does not affect the mathematical framework.

---

## Mathematical Framework

### State variables

Each urban sector `i` carries three variables at time `t`:

| Variable | Symbol | Layer | Semantics |
|---|---|---|---|
| Risk | `rᵢ ∈ [0,1]` | Systemic | Estimated severity of current conditions |
| Confidence | `cᵢ ∈ [0,1]` | Epistemic | Quality of situational awareness |
| Allocation | `aᵢ ∈ [0,1]` | Control | Fraction of sensing attention, `Σaᵢ = 1` |

Derived: `uᵢ = 1 − cᵢ` (uncertainty — the epistemic gap).

The separation of systemic and epistemic layers is not cosmetic. It reflects the core claim: *urban failures often arise not from high risk alone, but from the gap between risk and understanding.*

### Objective function

```
H(t) = Σᵢ  aᵢ(t) · rᵢ(t) · uᵢ(t)
```

**System entropy** measures risk-weighted residual uncertainty under current attention.

| H(t) state | Interpretation |
|---|---|
| H rising | Attention is misallocated; risk accumulates faster than awareness |
| H stable | System is maintaining situational awareness under pressure |
| H falling | Confidence is being built where risk is highest |

Policy design objective: **minimize cumulative H(t) over time**.

### State evolution

```
rᵢ(t+1) = rᵢ(t)·(1−δ)  +  Σⱼ→ᵢ wⱼᵢ·rⱼ(t)·uⱼ(t)²  −  μ·aᵢ(t)  +  eventᵢ(t)
cᵢ(t+1) = cᵢ(t)  +  η·aᵢ(t)·(1−cᵢ(t))  −  ρ·cᵢ(t)
```

Key design choices:

- **Spillover** `wⱼᵢ · rⱼ · uⱼ²`: risk *and* uncertainty must both be high for a sector to propagate — a low-risk sector generates little spillover even if unmonitored
- **Explicit Euler stepping**: spillover uses `rⱼ(t)`, not updated risk — no implicit self-feedback amplification within a step, making dynamics analytically tractable
- **Forgetting** `ρ·cᵢ`: structurally necessary — without it `cᵢ → 1`, `uᵢ → 0`, and the policy loses its ability to differentiate sectors
- **Event injection** `eventᵢ(t)`: how the perception layer (sensors, strikes, weather) enters the mathematical framework

### Allocation policy (v0.4)

```
aᵢ* ∝ exp(scoreᵢ / γ)      scoreᵢ = rᵢ·uᵢ + λ·outflowᵢ
```

where `outflowᵢ = Σⱼ wᵢⱼ · rᵢ · uᵢ²` is the marginal system value of attending to sector `i`.

The temperature parameter `γ` continuously controls allocation concentration:

| γ | Behavior |
|---|---|
| γ → 0 | Greedy: all attention on `argmax(score)` |
| γ = 0.5 | Balanced: moderate concentration |
| γ → ∞ | Uniform: Equal baseline |

### Key experimental finding

γ-scan experiments across four coupling environments (STABLE → EXPLOSIVE) show a strict monotone relationship: cumulative H(t) decreases as γ increases, with the minimum always at γ → ∞ in symmetric networks.

**Theoretical boundary result**:

> *Under persistent symmetric spillover with linear confidence dynamics, reactive entropy minimization converges to uniform allocation as the optimal strategy. Purely reactive policies — regardless of score design, floor heuristics, or regularization — cannot outperform uniform allocation in fully symmetric networks.*

This establishes the operating condition for ASAS advantage: **external event signals must break network symmetry** to justify non-uniform allocation. The system's value is in detecting and responding to that asymmetry.

See `docs/theory.md` and `benchmark/` for full derivations and experimental results.

---

## Pluggable Policies

All strategies implement one interface:

```python
class AllocationPolicy:
    def allocate(self, state: SystemState) -> Dict[str, float]:
        ...
```

| Policy | Formula | Notes |
|---|---|---|
| `EqualPolicy` | `aᵢ = 1/N` | γ→∞ limit; optimal under symmetric spillover |
| `RiskOnlyPolicy` | `aᵢ ∝ rᵢ` | Ignores epistemic state |
| `ReactivePolicy` | `aᵢ ∝ rᵢ·uᵢ` | Adds uncertainty awareness; with ε-floor |
| `SoftmaxPolicy` | `aᵢ ∝ exp(scoreᵢ/γ)` | Entropy-regularized; unifies all reactive policies |
| `AdaptivePolicy` | subclass | Base for stateful / predictive extensions (v0.5+) |

`SoftmaxPolicy` subsumes all others: `EqualPolicy` is γ→∞, `RiskOnlyPolicy` is γ→0 with `outflow_weight=0`.

---

## Experiments

```bash
# Policy comparison: reproduces v0.1 → v0.4 evolution
python benchmark/allocation_comparison.py
# → benchmark/policy_comparison.png

# Frankfurt strike scenario (no LLM required)
python examples/frankfurt_strike/scenario.py
python examples/frankfurt_strike/scenario.py --gamma 0.3 --steps 30

# With cognitive hub
ANTHROPIC_API_KEY=... python examples/frankfurt_strike/scenario.py --llm claude
GOOGLE_API_KEY=...   python examples/frankfurt_strike/scenario.py --llm gemini
```

---

## Quickstart

```python
from asas import ASASEngine
from asas.core.policy import SoftmaxPolicy

engine = ASASEngine.from_dict(
    sectors={
        "Hbf":     {"risk": 0.55, "confidence": 0.70},
        "Messe":   {"risk": 0.25, "confidence": 0.30},
        "Airport": {"risk": 0.35, "confidence": 0.45},
    },
    coupling={
        ("Messe", "Hbf"):     0.40,
        ("Messe", "Airport"): 0.30,
    },
    policy=SoftmaxPolicy(gamma=0.5),
)

engine.ingest({"Hbf": +0.40, "Airport": +0.15})  # inject event
state = engine.step()
print(f"H(t) = {engine.entropy:.3f}")

for sector, alloc in engine.priority_allocations():
    print(f"  {sector}: {alloc*100:.1f}%")
```

**With cognitive hub** (optional LLM reasoning):

```python
from asas.cognitive.claude import ClaudeHub

engine = ASASEngine.from_dict(..., cognitive_hub=ClaudeHub())
report = engine.analyze(context={"events": ["Ver.di strike active"]})
print(report.operational_report)
```

---

## Install

```bash
pip install -e .
```

Dependencies: `numpy`, `matplotlib`. LLM adapters: `anthropic` or `google-generativeai` (optional).

---

## Citation

```bibtex
@software{asas2026,
  title  = {ASAS: Active Situational Awareness System —
            A Control-Theoretic Framework for Urban Situational Awareness},
  year   = {2026},
  url    = {https://github.com/diwu0717/ASAS-Active-Situational-Awareness-System}
}
```

---
