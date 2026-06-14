# PR-HMC: Polar Reflective Hamiltonian Monte Carlo

## Overview

---

**Reflective Hamiltonian Monte Carlo (ReHMC)** samples from a distribution supported on a bounded domain by reflecting the Hamiltonian trajectory at the boundary of a feasible container. This repository implements a **convex-container-plus-thinning** approach for constrained targets. The main idea is to run reflective HMC on a simple smooth convex container and to use a smooth penalty to discourage trajectories from spending too much time outside the target support.

1. **Convex container.** The target support $K$ is embedded in a smooth convex container $\widetilde K$, such as a Euclidean or Frobenius ball. Reflections are performed only at the boundary of this container.
2. **Smooth penalty extension.** Outside $K$, the potential is modified by a smooth penalty 

   $$
   \tilde{U}(q) = U(q) + \lambda \psi\{\phi(q)/b\}.
   $$

   Here $\phi$ is a **smooth boundary-violation surrogate**: it is small on feasible points and increases as the point moves outside $K$. This provides a differentiable approximation to a hard support constraint and keeps the numerical dynamics more stable.

3. **Thinning / retention.** The Markov chain is simulated on the container $\widetilde K$. Samples that fall inside the original support $K$ are retained and used to estimate expectations under the original constrained target. For targets on the **Stiefel manifold** 

   $$
   V_{r,u} = \left\{ \Gamma \in \mathbb R^{r\times u} : \Gamma^\top \Gamma = I_u \right\},
   $$

    the sampler uses the **polar reparameterization** 

   $$
   \Gamma(B) = B(B^\top B)^{-1/2}.
   $$

    The matrix $B$ is restricted to the well-conditioned set 
   $$
   K = \left\{ B \in \mathbb R^{r\times u} : \lambda_{\min}(B^\top B) \ge c,\; \|B\|_F^2 \le M \right\},
   $$

    which is embedded in the Frobenius ball 
   $$
   \widetilde K = \left\{ B : \|B\|_F^2 \le M \right\}.
   $$

    This gives the **Polar Reflective HMC (PR-HMC)** sampler. The lower bound $\lambda_{\min}(B^\top B) \ge c$ prevents the polar map from becoming ill-conditioned, while the Frobenius ball provides a simple reflection boundary.

## How it works

---

The animation below illustrates the **container-based construction**. A chain runs reflective HMC inside the smooth convex container shown as the blue circle. The target support $K$ is shown in pink. Inside $K$, samples are retained. Outside $K$, the smooth penalty increases the potential and pushes the trajectory back toward the feasible region. At the container boundary, the trajectory reflects specularly.

![PR-HMC overview](assets/prhmc_overview.gif)

## Repository layout

---

```
src/ 
    polar_chmc.py       ConstrainedPolarHMC: PR-HMC for Stiefel-manifold targets using the polar reparameterization 
    general_chmc.py     GeneralConstrainedHMC: reflective HMC on a ball container for general constrained targets 
    dataset.py          Target log densities and data generators including mVMF and Bayesian PCA examples 
    utils.py            Step-size adaptation, mass adaptation, ESS, and subspace-alignment utilities

example_directional.py   PR-HMC on a matrix von Mises-Fisher distribution
example_pca.py           PR-HMC on a Bayesian PCA model with orthogonal loadings
example_polytope.py      PR-HMC on a polytope-constrained truncated Gaussian
```

## Installation

---

```bash
pip install -r requirements.txt
```

## Examples

---

Each example is self-contained; tuning constants are at the top of the file. The defaults are kept small so the scripts finish in a few minutes — increase `BURNIN`, `N_SAMPLES`, and the dimensions for paper-quality results (the paper uses `burnin = n_samples = 2000`, `4` chains, `50` leapfrog steps).

### 1. Directional distribution: matrix von Mises-Fisher

The first example samples from a matrix von Mises-Fisher distribution on $V_{p,k}$:

$$
\pi(\Gamma)
\propto
\exp \left\{\omega \mathrm{tr}(A^\top \Gamma)\right\},
\qquad
A =
\begin{bmatrix}
I_k \
0
\end{bmatrix}.
$$

The concentration parameter $\omega$ controls how strongly the distribution is concentrated around the leading coordinate directions.

```bash
python example_directional.py
```

```python
from src.polar_chmc import ConstrainedPolarHMC, run_chains
from src.dataset import make_A, logprob_fn_mvmf
import torch

p, k, omega = 20, 5, 100.0
A = make_A(p, k)

def make_sampler():
    return ConstrainedPolarHMC(
        logprob_fn=lambda q: logprob_fn_mvmf(q, A.to(q.dtype), p, k, kappa=omega),
        step_size=0.1, num_steps=50,
        use_reflection=True, use_soft_barrier=True,
        lam_penalty=2.0, rho=0.2, barrier_power=4,
        barrier_pos="softplus", target_accept=0.8,
    )

samples, budget, meta = run_chains(
    make_sampler, R=20.0, r=p, u=k, c=0.1, M=400.0,
    n_kept=2000, burnin=2000, n_chains=4, do_hungarian=True, verbose=False,
)
gamma = samples["Q"]   # (n_chains, n_kept, p, k) samples on the Stiefel manifold
```

### 2. Bayesian PCA with an orthogonal loading subspace

The second example considers a Bayesian PCA model with an orthogonal loading subspace. The loading matrix satisfies

$$
\Gamma \in V_{p,u}.
$$

The PR-HMC sampler samples the Stiefel-valued loading subspace through the polar representation.

```bash
python example_pca.py
```

```python
from src.polar_chmc import ConstrainedPolarHMC, run_chains
from src.dataset import simulate_latent_subspace, make_Phi_diag, logprob_ppca_macg
import torch

p, u, n, eps, beta = 10, 2, 30, 0.05, 1.0 / 100
R = torch.diag(torch.tensor([5.0, 0.5], dtype=torch.float64))
Y, A, Gamma0, W0 = simulate_latent_subspace(p=p, u=u, n=n, sigma_y=1.0, R=R, seed=123)
Phi = make_Phi_diag(p=p, m=u, eps=eps)

def make_sampler():
    return ConstrainedPolarHMC(
        logprob_fn=lambda q: logprob_ppca_macg(q, A.to(q.dtype), Phi.to(q.dtype), p, u, beta),
        step_size=0.1, num_steps=50,
        use_reflection=True, use_soft_barrier=True,
        lam_penalty=2.0, rho=0.2, barrier_power=4,
        barrier_pos="softplus", target_accept=0.8,
    )

samples, budget, meta = run_chains(
    make_sampler, R=15.0, r=p, u=u, c=0.1, M=200.0,
    n_kept=2000, burnin=2000, n_chains=4, do_hungarian=True, verbose=False,
)
```

### 3. Polytope-constrained Gaussian target

Truncated standard Gaussian on a polytope $K = \{x : A x \le b\}$ (e.g. the cube or the simplex), centered at the Chebyshev center.
The non-smooth boundary is handled by the smooth log-sum-exp and softplus boundary-violation surrogate inside the penalty term.

```bash
python example_polytope.py
```

```python
import numpy as np, torch
from src.general_chmc import GeneralConstrainedHMC
# see example_polytope.py for make_cube / chebyshev_center / PolytopeConstraint

A, b = make_cube(50)
x_c = chebyshev_center(A, b)
mu = torch.from_numpy(x_c)
constraint = PolytopeConstraint(A, b, R=...)

sampler = GeneralConstrainedHMC(
    logprob_fn=lambda q: -0.5 * ((q - mu.to(q.dtype)) ** 2).sum(),
    constraint=constraint, R=...,
    step_size=0.05, num_steps=50,
    use_reflection=True, use_soft_barrier=True,
    lam_penalty=2.0, rho=0.2, barrier_power=4, target_accept=0.65,
)
out = sampler.sample(d=50, q_init=mu.clone(), n_samples=2000, burnin=2000, verbose=False)
chain = out["chain"]   # (n_kept, d) samples inside the polytope
```

## Key hyperparameters

---

| Name              | Meaning                                                                  |
| ----------------- | ------------------------------------------------------------------------ |
| `R`             | radius of the ball container$\,\widetilde K$                           |
| `c`             | lower bound on$\,\lambda_{\min}(B^\top B)$ in the polar representation |
| `M`             | upper bound on$\|B\|_F^2$                                              |
| `lam_penalty`   | strength$\lambda$ of the smooth penalty outside $K$                  |
| `rho`           | boundary-layer width$b$ used in the penalty                            |
| `barrier_power` | penalty power in$\psi(z)=z^p$; the default is $p=4$                  |
| `barrier_pos`   | positive-part surrogate:`"softplus"`, `"hinge"`, or `"none"`       |
| `num_steps`     | number of leapfrog steps per HMC proposal                                |
| `step_size`     | leapfrog step size                                                       |
| `target_accept` | target acceptance rate used in step-size adaptation                      |

## Citation

---

```bibtex
@inproceedings{lee2026rehmc, 
    title = {Reflective Hamiltonian Monte Carlo: Mixing Analysis and Application to Sampling on the Stiefel Manifold}, 
    author = {Lee, Kwangmin and Park, Yeonhee and Park, Sewon}, 
    booktitle = {Proceedings of the 43rd International Conference on Machine Learning (ICML)}, 
    year = {2026}, 
}
```
