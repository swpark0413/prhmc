#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PR-HMC applied to Bayesian PCA with an orthogonal loading subspace.

Model (Section 6.2 of the paper):

    Y = Z R Gamma^T + E,    Gamma in V_{p,u}  (orthonormal loadings),

with latent scores Z, isotropic Gaussian noise E, and a diagonal scaling
matrix R = diag(5, 0.5) that resolves rotational non-identifiability. A
Matrix Angular Central Gaussian (MACG) prior is placed on Gamma via the
prior precision Phi = diag(1, eps); smaller eps yields a more anisotropic
(harder) posterior.

PR-HMC samples Gamma through the polar reparameterization on a constrained
Euclidean space, using reflective dynamics on a convex container plus
thinning.
"""
import numpy as np
import torch
import arviz as az

from src.polar_chmc import ConstrainedPolarHMC, run_chains
from src.dataset import simulate_latent_subspace, make_Phi_diag, logprob_ppca_macg


# ------------------------------------------------------------------
# Settings (paper uses p in {10,20,30,40}, u=2, n=30, eps in {0.1,0.05,0.01},
# burnin=2000, n_samples=2000, n_chains=4; reduced here for a quick demo)
# ------------------------------------------------------------------
P = 10            # ambient dimension r
U = 2             # subspace dimension
N_OBS = 30        # number of observations
EPS_ANISO = 0.05  # prior anisotropy (smaller = harder)
BETA = 1.0 / 100  # likelihood scaling
SIGMA_Y = 1.0     # observation noise
R_TRUE = [5.0, 0.5]

C_EIG = 0.1       # eigenvalue lower bound  lambda_min(B^T B) >= c
M_FRO = 200.0     # Frobenius bound  ||B||_F^2 <= M
R_BALL = 15.0     # convex container radius

BURNIN = 300
N_SAMPLES = 300
N_CHAINS = 4
NUM_LEAPFROG = 30
STEP_SIZE = 0.1
SEED = 0


def projection_diagnostics(Q):
    """Mean ESS and max split-Rhat over the entries of P = Gamma Gamma^T."""
    Q = Q.detach().cpu().numpy()
    C, T, p, u = Q.shape
    P = np.einsum("ctpr,ctqr->ctpq", Q, Q)
    iu = np.triu_indices(p)
    entries = P[:, :, iu[0], iu[1]]
    ess_list, rhat_list = [], []
    for j in range(entries.shape[-1]):
        x = entries[:, :, j]
        if np.allclose(x, x.flat[0]):
            continue
        idata = az.convert_to_dataset(x)
        ess_list.append(float(az.ess(idata).x.values))
        rhat_list.append(float(az.rhat(idata).x.values))
    return float(np.nanmean(ess_list)), float(np.nanmax(rhat_list))


def subspace_error(Q, Gamma0):
    """Mean projection (Frobenius) distance ||Gamma Gamma^T - G0 G0^T||_F."""
    Q = Q.detach().cpu().numpy().reshape(-1, Q.shape[-2], Q.shape[-1])
    G0 = Gamma0.detach().cpu().numpy()
    P0 = G0 @ G0.T
    dists = []
    for G in Q:
        if not np.isfinite(G).all():
            continue
        dists.append(np.linalg.norm(G @ G.T - P0, "fro"))
    return float(np.mean(dists))


def main():
    torch.set_num_threads(1)

    R_mat = torch.diag(torch.tensor(R_TRUE[:U], dtype=torch.float64))
    _, A, Gamma0, _ = simulate_latent_subspace(
        p=P, u=U, n=N_OBS, sigma_y=SIGMA_Y, R=R_mat, seed=123 + SEED
    )
    Phi = make_Phi_diag(p=P, m=U, eps=EPS_ANISO)

    def make_sampler():
        return ConstrainedPolarHMC(
            logprob_fn=lambda q: logprob_ppca_macg(
                q,
                A.to(dtype=q.dtype, device=q.device),
                Phi.to(dtype=q.dtype, device=q.device),
                P, U, BETA,
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

    print(f"[PR-HMC / Bayesian PCA] p={P}, u={U}, eps={EPS_ANISO}")
    samples, budget, meta = run_chains(
        make_sampler,
        R=R_BALL, r=P, u=U, c=C_EIG, M=M_FRO,
        n_kept=N_SAMPLES,
        burnin=BURNIN,
        thin=1,
        n_chains=N_CHAINS,
        seeds=[1 + 1000 * SEED, 2 + 1000 * SEED, 3 + 1000 * SEED, 4 + 1000 * SEED],
        verbose=False,
        kept_only=False,
        do_hungarian=True,
    )

    ess, rhat = projection_diagnostics(samples["Q"])
    err = subspace_error(samples["Q"], Gamma0)
    print(f"  acceptance (per chain) : {np.round(meta['acc_rate_per_chain'], 3)}")
    print(f"  mean ESS (projection)  : {ess:.1f}")
    print(f"  max split-Rhat         : {rhat:.4f}")
    print(f"  mean subspace error    : {err:.4f}")


if __name__ == "__main__":
    main()
