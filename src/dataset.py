import torch
from typing import Tuple
from src.utils import Gamma_from_B


def logprob_fn_mvmf(q, A: torch.Tensor, p: int, k: int, kappa: float = 1.0):

    X = q.view(p, k)
    A_ = A.to(dtype=X.dtype, device=X.device)

    U, _, Vh = torch.linalg.svd(X, full_matrices=False)
    Q = U @ Vh

    term_vonmises = kappa * torch.trace(A_.T @ Q)
    term_gauss = -0.5 * torch.sum(X * X)
    return term_gauss + term_vonmises


def logprob_fn_mbh(q, A: torch.Tensor, p: int, k: int):

    X = q.view(p, k)
    U, _, Vh = torch.linalg.svd(X, full_matrices=False)
    Q = U @ Vh

    term_bingham = -0.5 * torch.trace((Q.T @ A @ Q))

    term_gauss = -0.5 * torch.sum(X * X)
    return term_gauss + term_bingham

def sqexp_cov_and_precision(
    p: int,
    rho: float,
    *,
    return_precision: bool = True,
    jitter: float = 1e-6,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str = "cpu",
    verbose: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor | None]:


    device = torch.device(device)


    idx = torch.arange(p, dtype=dtype, device=device)
    dist = idx[:, None] - idx[None, :]


    Sigma = torch.exp(- (dist ** 2) / (2.0 * rho ** 2))

    if jitter > 0.0:
        Sigma = Sigma + jitter * torch.eye(p, dtype=dtype, device=device)


    if verbose:
        eigvals = torch.linalg.eigvalsh(Sigma)
        lam_min = eigvals.min()
        lam_max = eigvals.max()
        cond_Sigma = (lam_max / lam_min).item()

        print(f"[Sigma] λ_min = {lam_min:.3e}, λ_max = {lam_max:.3e}")
        print(f"[Sigma] condition number = {cond_Sigma:.3e}")

    if not return_precision:
        return Sigma, None

    A = torch.linalg.inv(Sigma)

    if verbose:
        eigvals_A = torch.linalg.eigvalsh(A)
        lam_min_A = eigvals_A.min()
        lam_max_A = eigvals_A.max()
        cond_A = (lam_max_A / lam_min_A).item()

        print(f"[A = Sigma^{-1}] λ_min = {lam_min_A:.3e}, λ_max = {lam_max_A:.3e}")
        print(f"[A] condition number = {cond_A:.3e}")

    return Sigma, A

def make_A(p, k):
    """Create target direction matrix A = [I_k; 0]"""
    M = torch.zeros((p, k))
    M[:k, :k] = torch.eye(k)
    return M


def simulate_latent_subspace(
    p: int,
    u: int,
    n: int,
    sigma_y=1.0,
    seed: int = 0,
    R: torch.Tensor | None = None,
    mu: torch.Tensor | None = None,
    *,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str = "cpu",
):

    device = torch.device(device)
    g = torch.Generator(device=device).manual_seed(seed)
    B0 = torch.randn(p, u, generator=g, device=device, dtype=dtype)
    Gamma0 = Gamma_from_B(B0).to(device=device, dtype=dtype)
    if R is None:
        R = torch.eye(u, device=device, dtype=dtype)
    else:
        R = R.to(device=device, dtype=dtype)
        if R.shape != (u, u):
            raise ValueError(f"R must have shape {(u,u)}, got {tuple(R.shape)}")

    W = Gamma0 @ R
    if mu is None:
        mu = torch.zeros(p, device=device, dtype=dtype)
    else:
        mu = mu.to(device=device, dtype=dtype)
        if mu.shape != (p,):
            raise ValueError(f"mu must have shape {(p,)}, got {tuple(mu.shape)}")

    X = torch.randn(u, n, generator=g, device=device, dtype=dtype)
    sigma_y_t = torch.as_tensor(sigma_y, device=device, dtype=dtype)
    E = sigma_y_t * torch.randn(p, n, generator=g, device=device, dtype=dtype)
    Y = (W @ X) + mu[:, None] + E
    A = Y @ Y.T

    return Y, A, Gamma0, W


def make_Phi_diag(p: int, m: int, eps: float) -> torch.Tensor:

    d = torch.full((p,), eps, dtype=torch.float64)
    d[:m] = 1.0
    return torch.diag(d)


def logprob_ppca_macg(q: torch.Tensor, A: torch.Tensor, Phi: torch.Tensor, p: int, u: int, beta: float) -> torch.Tensor:
    B = q.view(p, u)
    Gamma = Gamma_from_B(B)
    dtype  = B.dtype
    device = B.device

    Phi = Phi.to(device=device, dtype=dtype)
    Gamma   = Gamma.to(device=device, dtype=dtype)
    A       = A.to(device=device, dtype=dtype)

    Phi_inv = torch.linalg.inv(Phi)
    term_prior = -0.5 * torch.trace(B.T @ Phi_inv @ B)
    term_like = 0.5 * beta * torch.trace(Gamma.T @ A @ Gamma)

    return term_prior + term_like
