# Drucker-Prager Plasticity: Theory and Implementation Guide

## 1. Introduction

The Drucker-Prager (DP) yield criterion is a pressure-dependent plasticity model widely used for geomaterials (soil, rock, concrete). Unlike the von Mises (J2) model, DP accounts for the influence of hydrostatic pressure on yielding.

---

## 2. The Drucker-Prager Yield Criterion

### Standard Formulation

$$f = \sqrt{J_2} + \alpha I_1 - k = 0$$

Where:
- $I_1 = \text{tr}(\boldsymbol{\sigma}) = \sigma_{11} + \sigma_{22} + \sigma_{33}$ — first stress invariant (hydrostatic)
- $J_2 = \frac{1}{2}\mathbf{s}:\mathbf{s}$ — second invariant of deviatoric stress
- $\mathbf{s} = \boldsymbol{\sigma} - \frac{I_1}{3}\mathbf{I}$ — deviatoric stress tensor
- $\alpha, k$ — material parameters (friction and cohesion related)

### Geometric Interpretation

In principal stress space, the DP yield surface forms a **cone** with:
- Axis along the hydrostatic line ($\sigma_1 = \sigma_2 = \sigma_3$)
- Apex (tip) at $I_1 = k/\alpha$
- Opening angle controlled by $\alpha$

---

## 3. The Apex Singularity Problem

### What is the Problem?

At the cone apex, the **gradient of the yield function is undefined**. This causes:

1. **Numerical instability**: Return mapping algorithm fails to converge
2. **Non-differentiability**: JAX auto-differentiation produces NaN
3. **Multi-valued flow direction**: Plastic flow direction becomes discontinuous

### Mathematical Description

The plastic flow direction is:
$$\mathbf{n} = \frac{\partial f}{\partial \boldsymbol{\sigma}} = \frac{\mathbf{s}}{2\sqrt{J_2}} + \frac{\alpha}{3}\mathbf{I}$$

When $J_2 \rightarrow 0$ (approaching apex), $\frac{\mathbf{s}}{2\sqrt{J_2}} \rightarrow \text{undefined}$.

---

## 4. Solution: Hyperbolic Regularization

### The Idea

Replace the sharp cone apex with a smooth hyperbolic surface that is everywhere differentiable:

$$f_{reg} = \sqrt{J_2 + a^2} + \alpha I_1 - k = 0$$

Where $a > 0$ is a small regularization parameter.

### Why This Works

1. $\sqrt{J_2 + a^2}$ is always positive and differentiable
2. When $J_2 \gg a^2$: behaves like standard DP
3. When $J_2 \approx 0$: smooth transition at apex
4. The regularized gradient: $\frac{\mathbf{s}}{2\sqrt{J_2 + a^2}}$ is always defined

### Choosing Parameter $a$

- Too large: deviates from true DP behavior
- Too small: may still have numerical issues
- **Recommended**: $a \approx 0.001 \cdot k$ to $0.01 \cdot k$

---

## 5. Return Mapping Algorithm

For a trial stress $\boldsymbol{\sigma}_{trial}$ that violates yield ($f > 0$):

### Step 1: Compute Trial Quantities
```
I1_trial = trace(sigma_trial)
s_trial = sigma_trial - (I1_trial/3) * I
J2_trial = 0.5 * s_trial : s_trial
```

### Step 2: Check Yield (Regularized)
```
f = sqrt(J2_trial + a^2) + alpha * I1_trial - k
```

### Step 3: If f > 0, Compute Plastic Correction

The return mapping for DP with associated flow:
```
sqrt_J2_reg = sqrt(J2_trial + a^2)
delta_lambda = f / (1 + 3*alpha^2)  # Consistency parameter
sigma = sigma_trial - delta_lambda * (s_trial/(2*sqrt_J2_reg) + alpha*I)
```

### Step 4: Handle Apex Region

If the return point lies beyond the apex (I1 > k/alpha), return to apex:
```
sigma_apex = (k / (3*alpha)) * I
```

---

## 6. Relating DP Parameters to Physical Properties

### From Mohr-Coulomb Parameters

For matching with Mohr-Coulomb (friction angle $\phi$, cohesion $c$):

**Inscribed cone (compression)**:
$$\alpha = \frac{2\sin\phi}{\sqrt{3}(3-\sin\phi)}, \quad k = \frac{6c\cos\phi}{\sqrt{3}(3-\sin\phi)}$$

**Circumscribed cone (tension)**:
$$\alpha = \frac{2\sin\phi}{\sqrt{3}(3+\sin\phi)}, \quad k = \frac{6c\cos\phi}{\sqrt{3}(3+\sin\phi)}$$

### Example Values

| Material | $\phi$ (°) | $c$ (MPa) | $\alpha$ | $k$ (MPa) |
|----------|-----------|-----------|----------|-----------|
| Dense sand | 35 | 0 | 0.29 | 0 |
| Concrete | 37 | 3.0 | 0.31 | 7.6 |
| Rock | 45 | 10.0 | 0.38 | 22.8 |

---

## 7. Code Structure Overview

The implementation in `drucker_prager_plasticity.py` follows this structure:

1. **Material parameters**: Define elastic (E, ν) and DP (α, k, a) constants
2. **DruckerPragerPlasticity class**: Inherits from JAX-FEM Problem class
   - `custom_init()`: Initialize stress/strain history
   - `get_tensor_map()`: Return the stress computation function
   - `get_maps()`: Define strain, elastic stress, and DP return mapping
   - `update_stress_strain()`: Update internal variables after each step
3. **Main block**: Set up mesh, BCs, and run incremental loading

---

## 8. References

1. Drucker, D.C. and Prager, W. (1952). "Soil mechanics and plastic analysis or limit design." *Quarterly of Applied Mathematics*, 10(2), 157-165.

2. Simo, J.C. and Hughes, T.J.R. (2006). *Computational Inelasticity*. Springer.

3. de Souza Neto, E.A., Peric, D., and Owen, D.R.J. (2008). *Computational Methods for Plasticity*. Wiley.
