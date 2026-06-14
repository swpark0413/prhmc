#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PR-HMC applied to a polytope-constrained target (Appendix D of the paper).

A polytope K = {x in R^d : A x <= b} has a non-smooth (piecewise-linear)
boundary. PR-HMC embeds K into a convex Frobenius/Euclidean ball container,
replaces the non-smooth constraint by a smooth log-sum-exp + softplus
surrogate inside the penalty, runs reflective HMC on the ball, and recovers
samples in K by thinning.

Target: truncated standard Gaussian centered at the Chebyshev center of K,

    pi(x) propto exp(-||x - x_c||^2 / 2) * 1{x in K}.

This script applies PR-HMC to the d-cube and the d-simplex (no comparison
against other samplers).
"""
import numpy as np
import torch
import arviz as az
from scipy.optimize import linprog

from src.general_chmc import GeneralConstrainedHMC


# ------------------------------------------------------------------
# Settings (paper uses d=100, burnin=2000, n_samples=2000, n_chains=4;
# reduced here for a quick demo)
# ------------------------------------------------------------------
DIM = 50
BURNIN = 500
N_SAMPLES = 500
N_CHAINS = 4
NUM_LEAPFROG = 50
STEP_SIZE = 0.05


def make_cube(d):
    """K = [-1, 1]^d."""
    A = np.vstack([np.eye(d), -np.eye(d)])
    b = np.ones(2 * d)
    return A, b


def make_simplex(d):
    """K = {x >= 0, sum(x) <= 1}."""
    A = np.vstack([-np.eye(d), np.ones((1, d))])
    b = np.concatenate([np.zeros(d), [1.0]])
    return A, b


def chebyshev_center(A, b):
    """Center of the largest inscribed ball, via a linear program."""
    m, d = A.shape
    norms = np.linalg.norm(A, axis=1)
    c = np.zeros(d + 1)
    c[-1] = -1.0
    A_lp = np.hstack([A, norms.reshape(-1, 1)])
    res = linprog(c, A_ub=A_lp, b_ub=b, method="highs")
    return res.x[:d] if res.success else np.zeros(d)


class PolytopeConstraint:
    """Wrapper for {A x <= b} consumed by GeneralConstrainedHMC."""

    def __init__(self, A, b, R):
        self.A_t = torch.from_numpy(A.astype(np.float64))
        self.b_t = torch.from_numpy(b.astype(np.float64))
        self.d = A.shape[1]
        self.m = A.shape[0]
        self._R = float(R)

    def violations_vec(self, q):
        A = self.A_t.to(device=q.device, dtype=q.dtype)
        b = self.b_t.to(device=q.device, dtype=q.dtype)
        return A @ q - b

    def violations(self, q):
        v = self.violations_vec(q)
        return [v[i] for i in range(self.m)]

    def in_K(self, q):
        A = self.A_t.to(device=q.device, dtype=q.dtype)
        b = self.b_t.to(device=q.device, dtype=q.dtype)
        return bool((A @ q <= b + 1e-10).all().item())

    def container_radius(self):
        return self._R


def run_polytope(name, A, b):
    d = A.shape[1]
    x_c = chebyshev_center(A, b)
    mu = torch.from_numpy(x_c.astype(np.float64))

    # circumscribing-ball radius for the container
    norms_A = np.linalg.norm(A, axis=1)
    slacks = (b - A @ x_c) / (norms_A + 1e-30)
    R = float(np.linalg.norm(x_c)) + float(np.max(slacks))

    constraint = PolytopeConstraint(A, b, R)

    def logprob_fn(q):
        diff = q - mu.to(dtype=q.dtype, device=q.device)
        return -0.5 * (diff * diff).sum()

    q_init = mu.clone()
    chains = []
    for ch in range(N_CHAINS):
        torch.manual_seed(42 + ch)
        sampler = GeneralConstrainedHMC(
            logprob_fn=logprob_fn,
            constraint=constraint,
            R=R,
            step_size=STEP_SIZE,
            num_steps=NUM_LEAPFROG,
            use_reflection=True,
            use_soft_barrier=True,
            lam_penalty=2.0,
            rho=0.2,
            barrier_sigma=10.0,
            barrier_power=4,
            target_accept=0.65,
        )
        out = sampler.sample(
            d=d, q_init=q_init.clone(),
            n_samples=N_SAMPLES, burnin=BURNIN,
            verbose=False, max_total_steps=BURNIN + N_SAMPLES * 5,
        )
        chains.append(out["chain"].numpy())

    min_len = min(c.shape[0] for c in chains)
    chains = np.stack([c[:min_len] for c in chains], axis=0)  # (C, T, d)

    # per-coordinate ESS / split-Rhat
    idata = az.convert_to_dataset(chains)
    ess = float(np.nanmean(az.ess(idata).x.values))
    rhat = float(np.nanmax(az.rhat(idata).x.values))

    print(f"[PR-HMC / {name}] d={d}")
    print(f"  kept samples per chain : {min_len}")
    print(f"  mean ESS (coordinates) : {ess:.1f}")
    print(f"  max split-Rhat         : {rhat:.4f}")


def main():
    torch.set_num_threads(1)
    A, b = make_cube(DIM)
    run_polytope(f"{DIM}-cube", A, b)
    A, b = make_simplex(DIM)
    run_polytope(f"{DIM}-simplex", A, b)


if __name__ == "__main__":
    main()
