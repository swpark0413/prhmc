import math
import torch
from scipy.optimize import linear_sum_assignment


def init_polar(p, k, eps=0.05, *, device="cpu", dtype=torch.float64):
    X = torch.randn(p, k, dtype=dtype, device=device)
    U, _, Vh = torch.linalg.svd(X, full_matrices=False)
    Q = U @ Vh
    Z = eps * torch.randn(p, k, dtype=dtype, device=device)
    return (Q + Z).reshape(-1)

def Gamma_from_B(B: torch.Tensor) -> torch.Tensor:
        U, _, Vh = torch.linalg.svd(B, full_matrices=False)
        return U @ Vh


class DualAveragingStepSize:
    def __init__(
        self,
        init_step_size: float,
        target_accept: float = 0.8,
        gamma: float = 0.05,
        t0: float = 10.0,
        kappa: float = 0.75,
    ):
        self.target = float(target_accept)
        self.gamma = float(gamma)
        self.t0 = float(t0)
        self.kappa = float(kappa)


        self.mu = math.log(10.0 * init_step_size)

        self.hbar = 0.0
        self.log_eps = math.log(init_step_size)
        self.log_eps_bar = math.log(init_step_size)
        self.t = 0

    def update(self, accept_prob: float) -> float:
        self.t += 1
        t = self.t

        eta = 1.0 / (t + self.t0)
        self.hbar = (1.0 - eta) * self.hbar + eta * (self.target - accept_prob)

        self.log_eps = self.mu - (math.sqrt(t) / self.gamma) * self.hbar

        w = t ** (-self.kappa)
        self.log_eps_bar = w * self.log_eps + (1.0 - w) * self.log_eps_bar

        return math.exp(self.log_eps)

    def final_step_size(self) -> float:
        return math.exp(self.log_eps_bar)


@torch.no_grad()
def make_mass_diag_from_var(
    var: torch.Tensor,
    jitter: float = 1e-3,
    min_m: float = 1e-6,
    max_m: float = 1e6,
) -> torch.Tensor:
    mass = var + jitter
    mass = torch.clamp(mass, min=min_m, max=max_m)
    return mass

class RunningDiagVar:

    def __init__(self, d: int, device=None, dtype=None):
        self.n = 0
        self.mean = torch.zeros(d, device=device, dtype=dtype)
        self.M2 = torch.zeros(d, device=device, dtype=dtype)

    @torch.no_grad()
    def update(self, x: torch.Tensor):

        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.M2 += delta * delta2

    @torch.no_grad()
    def var(self) -> torch.Tensor:
        if self.n < 2:
            return torch.ones_like(self.mean)
        return self.M2 / (self.n - 1)


def align_perm_sign_hungarian(Q: torch.Tensor, Qref: torch.Tensor) -> torch.Tensor:

    C = Qref.T @ Q
    Cabs = torch.abs(C).detach().cpu().numpy()
    row_ind, col_ind = linear_sum_assignment(-Cabs)

    Qp = Q[:, col_ind]
    s = torch.sign(torch.diag(Qref.T @ Qp))
    s[s == 0] = 1.0
    return Qp * s

def procrustes_align(Q: torch.Tensor, Qref: torch.Tensor) -> torch.Tensor:
    M = Qref.T @ Q
    U, _, Vh = torch.linalg.svd(M)
    R = U @ Vh
    return Q @ R.T

def align_to_ref_hungarian_then_procrustes(Q: torch.Tensor, Qref: torch.Tensor) -> torch.Tensor:
    Q1 = align_perm_sign_hungarian(Q, Qref)
    Q2 = procrustes_align(Q1, Qref)
    return Q2

def align_filled_for_rhat(Q_post: torch.Tensor) -> torch.Tensor:

    C, T, p, r = Q_post.shape
    Qref = Q_post[0, 0]
    out = torch.empty_like(Q_post)
    for c in range(C):
        for t in range(T):
            out[c, t] = align_to_ref_hungarian_then_procrustes(Q_post[c, t], Qref)
    return out













def nanmin(x, dim):
    mask = ~torch.isnan(x)
    x2 = torch.where(mask, x, torch.full_like(x, float("inf")))
    return x2.min(dim=dim).values

def nanmax(x, dim):
    mask = ~torch.isnan(x)
    x2 = torch.where(mask, x, torch.full_like(x, float("-inf")))
    return x2.max(dim=dim).values

def nanmean(x, dim):
    mask = ~torch.isnan(x)
    x2 = torch.where(mask, x, torch.zeros_like(x))
    cnt = mask.sum(dim=dim).clamp(min=1)
    return x2.sum(dim=dim) / cnt
