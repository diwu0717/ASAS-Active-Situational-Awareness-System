# ASAS v0.4 — Abstract

**ASAS: A Control-Theoretic Framework for Urban Situational Awareness
with Entropy-Regularized Attention Allocation**

---

## Abstract

We present ASAS (Active Situational Awareness System), a control-theoretic
framework for urban situational awareness that formulates attention allocation
as a dynamic optimization problem over a network of coupled sectors.
Conventional monitoring systems treat risk signals independently and reactively,
ignoring spillover dynamics between domains and the fact that epistemic
uncertainty is itself controllable through directed attention. ASAS addresses
this limitation by explicitly modeling each urban sector with two state layers:
systemic risk $r_i$, representing what is happening, and epistemic confidence
$c_i$, representing what the system knows. Attention allocation $a_i$
is modeled as the sole control variable satisfying a simplex constraint.

We define a risk-weighted uncertainty functional

$$H(t) = \sum_i a_i(t) \, r_i(t) \, u_i(t), \quad u_i = 1 - c_i,$$

which measures the residual epistemic gap under current allocation.
State evolution follows explicit-Euler dynamics in which risk propagates
between sectors as $w_{ij} \, r_j(t) \, u_j(t)^2$, encoding the principle
that spillover requires both systemic risk and epistemic uncertainty.

We derive an entropy-regularized softmax allocation policy

$$a_i^* \propto \exp(\mathrm{score}_i / \gamma), \quad
\mathrm{score}_i = r_i u_i + \lambda \, \mathrm{outflow}_i,$$

via constrained optimization with temperature parameter $\gamma$.
This formulation unifies reactive attention strategies into a single
two-parameter family that interpolates between greedy concentration
($\gamma \to 0$) and uniform allocation ($\gamma \to \infty$).

Through $\gamma$-scan experiments across four coupling regimes (stable,
marginal, unstable, explosive), we establish a theoretical boundary result:
under persistent symmetric spillover and linear confidence dynamics,
any reactive entropy-minimizing policy converges to uniform allocation,
independent of $\gamma$. We formalize this as a symmetry invariance
proposition and identify its breaking condition — network asymmetry induced
by external event injection — as the domain in which ASAS provides advantage
over uniform monitoring.

Experimental results on a five-sector urban network confirm the symmetry
invariance proposition and demonstrate ASAS advantage under asymmetric
event injection. The framework is implemented as an open-source Python
library with a modular architecture separating state representation,
dynamics, objective, policy, and engine. A plug-in cognitive interface
enables optional LLM-based situational reporting without coupling the
mathematical core to a specific model.

**Keywords**: situational awareness, attention allocation, entropy minimization,
network spillover dynamics, epistemic uncertainty, control theory, urban systems
