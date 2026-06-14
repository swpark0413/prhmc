#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PR-HMC applied to a directional distribution on the Stiefel manifold.

Target: Matrix von Mises-Fisher (mVMF) distribution on V_{p,k},

    pi(Gamma) propto exp( omega * tr(A^T Gamma) ),   Gamma^T Gamma = I_k,

with A = [I_k ; 0] so that the mass concentrates around the canonical
k-dimensional subspace, and omega > 0 controlling the concentration
(Section 6.1 of the paper).

PR-HMC samples through the polar reparameterization Gamma(B) = B (B^T B)^{-1/2}
on the constrained Euclidean space, using reflective dynamics on a convex
container plus thinning.
"""
import numpy as np
import torch
import arviz as az

from src.polar_chmc import ConstrainedPolarHMC, run_chains
from src.dataset import make_A, logprob_fn_mvmf


# ------------------------------------------------------------------
# Settings (paper uses p in {20,40}, k up to p-1, omega up to 500,
# burnin=2000, n_samples=2000, n_chains=4; reduced here for a quick demo)
# ------------------------------------------------------------------
P = 20            # ambient dimension r
K = 5             # subspace dimension u
OMEGA = 100.0     # concentration
C_EIG = 0.1       # eigenvalue lower bound  lambda_min(B^T B) >= c
M_FRO = 400.0     # Frobenius bound  ||B||_F^2 <= M
R_BALL = 20.0     # convex container radius

BURNIN = 300
N_SAMPLES = 300
N_CHAINS = 4
NUM_LEAPFROG = 30
STEP_SIZE = 0.1


def projection_diagnostics(Q):
    """ESS / split-Rhat on the entries of the projection matrix Gamma Gamma^T.

    Q: (C, T, p, k) tensor of sampled Stiefel matrices.
    The projection P = Gamma Gamma^T is a rotation-invariant summary of the
    sampled subspace; we report the mean ESS and the max split-Rhat over its
    upper-triangular entries.
    """
    Q = Q.detach().cpu().numpy()
    C, T, p, k = Q.shape
    P = np.einsum("ctpr,ctqr->ctpq", Q, Q)  # (C, T, p, p)
    iu = np.triu_indices(p)
    entries = P[:, :, iu[0], iu[1]]         # (C, T, n_entries)

    ess_list, rhat_list = [], []
    for j in range(entries.shape[-1]):
        x = entries[:, :, j]                # (C, T) = (chain, draw)
        if np.allclose(x, x.flat[0]):
            continue
        idata = az.convert_to_dataset(x)
        ess_list.append(float(az.ess(idata).x.values))
        rhat_list.append(float(az.rhat(idata).x.values))
    return float(np.nanmean(ess_list)), float(np.nanmax(rhat_list))


def main():
    torch.set_num_threads(1)
    A = make_A(P, K)  # A = [I_k ; 0], shape (p, k)

    def make_sampler():
        return ConstrainedPolarHMC(
            logprob_fn=lambda q: logprob_fn_mvmf(
                q, A.to(dtype=q.dtype, device=q.device), P, K, kappa=OMEGA
            ),
            step_size=STEP_SIZE,
            num_steps=NUM_LEAPFROG,
            device="cpu",
            dtype=torch.float64,
            use_reflection=True,
            use_soft_barrier=True,
            beta_phiK=10.0,
            lam_penalty=2.0,
            rho=0.2,
            barrier_sigma=10.0,
            barrier_power=4,
            barrier_pos="softplus",
            hit_time=True,
            target_accept=0.8,
        )

    print(f"[PR-HMC / mVMF] V_{{{P},{K}}}, omega={OMEGA}")
    samples, budget, meta = run_chains(
        make_sampler,
        R=R_BALL, r=P, u=K, c=C_EIG, M=M_FRO,
        n_kept=N_SAMPLES,
        burnin=BURNIN,
        thin=1,
        n_chains=N_CHAINS,
        verbose=False,
        kept_only=False,
        do_hungarian=True,
    )

    ess, rhat = projection_diagnostics(samples["Q"])
    print(f"  acceptance (per chain): {np.round(meta['acc_rate_per_chain'], 3)}")
    print(f"  mean ESS (projection)  : {ess:.1f}")
    print(f"  max split-Rhat         : {rhat:.4f}")


if __name__ == "__main__":
    main()
