# Mathematical Foundation of ASAS

## 1. Variables

Risk_i ∈ [0,1]  
Confidence_i ∈ [0,1]  
Uncertainty_i = 1 - Confidence_i  

Allocation_i ∈ [0,1], with Σ Allocation_i = 1  

---

## 2. Allocation Function

A_i = α R_i (1 - C_i) + β R_i C_i  

Where:

α = exploration weight  
β = stabilization weight  

Normalized allocation:

w_i = A_i / Σ A_j  

---

## 3. Residual Entropy

E_t = Σ w_i × (1 - C_i)

Objective:

Minimize E_t over time.

---

## 4. Dynamic Update Rule

C_i(t+1) = C_i(t) + η w_i (1 - C_i(t))

η = learning rate

This ensures that higher allocation accelerates confidence gain.
