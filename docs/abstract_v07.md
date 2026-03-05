# ASAS v0.7 — Abstract

**ASAS: A Control-Theoretic Framework for Urban Situational Awareness
with Model Predictive Attention Allocation**

---

## Abstract

We present ASAS (Active Situational Awareness System), a control-theoretic
framework for urban situational awareness that formulates attention allocation
as a dynamic optimization problem over a network of coupled sectors.
Conventional monitoring systems treat risk signals independently and reactively,
ignoring spillover dynamics between domains and the fact that epistemic
uncertainty is itself controllable through directed attention. ASAS addresses
this by representing each sector with two state layers — systemic risk $r_i$
and epistemic confidence $c_i$ — with attention allocation $a_i$ as the sole
control variable subject to a simplex constraint.

We define a risk-weighted uncertainty functional

$$H(t) = \sum_i a_i(t)\, r_i(t)\, u_i(t), \quad u_i = 1 - c_i,$$

and derive state evolution dynamics in which risk propagates between sectors
as $w_{ij} r_j u_j^2$, encoding the principle that spillover requires both
systemic risk and epistemic uncertainty at the source.

Through a systematic policy evolution from reactive allocation (v0.4) to
one-step predictive control (v0.5), we establish a theoretical boundary
result: under persistent symmetric spillover, any reactive policy converges
to uniform allocation, independent of score design or temperature parameter.
We formalize this as a symmetry invariance proposition and identify its
breaking condition — network asymmetry induced by external event injection —
as the domain in which intelligent allocation provides advantage.

We then show that one-step predictive policies (v0.5) fail to break this
boundary due to confidence asymmetry distortion and risk saturation masking,
and introduce the ASAS-MPC Policy (v0.7): a closed-loop model predictive
control formulation that evaluates N candidate allocations via T-step rollout
under a conservative baseline, at $O(N^2 T)$ complexity per step.

Horizon sensitivity experiments over $T \in \{1,\ldots,20\}$ demonstrate
that the improvement over uniform allocation $\Delta(T)$ is strictly
monotonically increasing in $T$ with no optimal finite horizon, reaching
14.4\% at $T=1$ and 22.8\% at $T=20$. The efficiency frontier analysis
identifies $T=5$ as the practical optimum: 15.7\% improvement at 5.6ms
per step. MPC beats uniform allocation across all four tested coupling
environments (stable through explosive), with advantage scaling monotonically
with coupling strength.

The framework is implemented as an open-source Python library with a
modular five-layer architecture and an optional LLM cognitive interface.
A reference deployment scenario demonstrates the system on the Feb 2,
2026 Frankfurt transport strike.

**Keywords**: situational awareness, attention allocation, entropy minimization,
network spillover dynamics, model predictive control, epistemic uncertainty,
urban systems
