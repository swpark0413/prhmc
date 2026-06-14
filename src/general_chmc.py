"""
General Constrained HMC with ball container + thinning.
Supports arbitrary constraint sets K embedded in a ball container K_tilde.
"""
from typing import Callable, Optional, Dict, Any, List
import torch

from src.utils import DualAveragingStepSize, RunningDiagVar, make_mass_diag_from_var


class AnnulusConstraint:
    """K = {x in R^d : a <= ||x|| <= b}  (nonconvex)"""

    def __init__(self, a: float, b: float):
        self.a = float(a)
        self.b = float(b)

    def violations(self, q: torch.Tensor) -> List[torch.Tensor]:
        """Return list of scalar violation terms, each <= 0 iff feasible."""
        norm_sq = (q * q).sum()
        v1 = self.a ** 2 - norm_sq
        v2 = norm_sq - self.b ** 2
        return [v1, v2]

    def violations_vec(self, q: torch.Tensor) -> torch.Tensor:
        """Vectorized: return (2,) tensor."""
        norm_sq = (q * q).sum()
        return torch.stack([self.a ** 2 - norm_sq, norm_sq - self.b ** 2])

    def in_K(self, q: torch.Tensor) -> bool:
        norm_sq = (q * q).sum().item()
        return (self.a ** 2 <= norm_sq) and (norm_sq <= self.b ** 2)

    def container_radius(self) -> float:
        return self.b


class SimplexConstraint:
    """K = {x in R^d : x_i >= 0, sum(x) <= 1}"""

    def __init__(self, d: int, R_container: Optional[float] = None):
        self.d = d

        self._R = R_container if R_container is not None else 2.0

    def violations(self, q: torch.Tensor) -> List[torch.Tensor]:
        """Each -x_i and sum(x)-1, all <= 0 iff feasible."""
        vs = []
        for i in range(self.d):
            vs.append(-q[i])
        vs.append(q.sum() - 1.0)
        return vs

    def violations_vec(self, q: torch.Tensor) -> torch.Tensor:
        """Vectorized: return (d+1,) tensor."""
        return torch.cat([-q, (q.sum() - 1.0).unsqueeze(0)])

    def in_K(self, q: torch.Tensor) -> bool:
        return bool((q >= -1e-12).all().item() and (q.sum().item() <= 1.0 + 1e-12))

    def container_radius(self) -> float:
        return self._R


class GeneralConstrainedHMC:
    """
    Reflective HMC on a ball container K_tilde with penalty-based
    extended potential and thinning to recover samples from K.
    """

    def __init__(
        self,
        logprob_fn: Callable[[torch.Tensor], torch.Tensor],
        constraint,
        *,
        R: Optional[float] = None,
        step_size: float = 0.05,
        num_steps: int = 50,
        device: str = "cpu",
        dtype: torch.dtype = torch.float64,


        use_reflection: bool = True,
        use_soft_barrier: bool = True,
        beta_phiK: float = 10.0,
        lam_penalty: float = 10.0,
        rho: float = 0.2,
        barrier_sigma: float = 10.0,
        barrier_power: int = 4,


        max_reflections: int = 50,
        eps: float = 1e-12,


        target_accept: float = 0.65,
        adapt_step_size: bool = True,
        adapt_mass: bool = True,
        eps_min: float = 1e-6,
        eps_max: float = 1.0,
        mass_jitter: float = 1e-3,
        mass_min: float = 1e-6,
        mass_max: float = 1e6,
    ):
        self.logprob_fn = logprob_fn
        self.constraint = constraint
        self.R = R if R is not None else constraint.container_radius()
        self.step_size = float(step_size)
        self.num_steps = int(num_steps)
        self.device = device
        self.dtype = dtype

        self.use_reflection = use_reflection
        self.use_soft_barrier = use_soft_barrier
        self.beta_phiK = float(beta_phiK)
        self.lam = float(lam_penalty)
        self.rho = float(rho)
        self.barrier_sigma = float(barrier_sigma)
        self.barrier_power = int(barrier_power)

        self.max_reflections = int(max_reflections)
        self.eps = float(eps)

        self.target_accept = float(target_accept)
        self.adapt_step_size = adapt_step_size
        self.adapt_mass = adapt_mass
        self.eps_min = float(eps_min)
        self.eps_max = float(eps_max)
        self.mass_jitter = float(mass_jitter)
        self.mass_min = float(mass_min)
        self.mass_max = float(mass_max)

        self.mass_diag: Optional[torch.Tensor] = None


    def U_base(self, q: torch.Tensor) -> torch.Tensor:
        return -self.logprob_fn(q)


    def phi_pos_softplus(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.softplus(self.barrier_sigma * x) / self.barrier_sigma

    def psi(self, z: torch.Tensor) -> torch.Tensor:
        return z ** self.barrier_power

    def U_tilde(self, q: torch.Tensor) -> torch.Tensor:
        U0 = self.U_base(q)
        if not self.use_soft_barrier:
            return U0


        if hasattr(self.constraint, 'violations_vec'):
            V = self.constraint.violations_vec(q)
            pos = torch.nn.functional.softplus(self.barrier_sigma * V) / self.barrier_sigma
            pos_sum = pos.sum()
        else:

            vs = self.constraint.violations(q)
            if len(vs) <= 50:
                pos_sum = sum(self.phi_pos_softplus(v) for v in vs)
            else:
                V = torch.stack(vs)
                pos = torch.nn.functional.softplus(self.barrier_sigma * V) / self.barrier_sigma
                pos_sum = pos.sum()

        z = pos_sum / self.rho
        return U0 + self.lam * self.psi(z)

    def grad_U_tilde(self, q: torch.Tensor) -> torch.Tensor:
        q_req = q.clone().detach().requires_grad_(True)
        Uv = self.U_tilde(q_req)
        (gq,) = torch.autograd.grad(Uv, q_req, create_graph=False)
        return gq.detach()


    def phi_container(self, q: torch.Tensor) -> float:
        return (q * q).sum().item() - self.R ** 2

    def _project_to_ball(self, q: torch.Tensor) -> torch.Tensor:
        n = q.norm()
        if n.item() <= self.R:
            return q
        return q * (self.R / (n + self.eps))

    def _outward_normal(self, q: torch.Tensor) -> torch.Tensor:
        return q / (q.norm() + self.eps)

    def _reflect_momentum(self, p: torch.Tensor, n_hat: torch.Tensor,
                          mass_diag: torch.Tensor) -> torch.Tensor:
        v = p / mass_diag
        v_ref = v - 2.0 * (v @ n_hat) * n_hat
        return mass_diag * v_ref

    def _hit_time_ball(self, q: torch.Tensor, v: torch.Tensor) -> float:
        a = (v @ v).item()
        b = 2.0 * (q @ v).item()
        c = (q @ q).item() - self.R ** 2
        if a <= self.eps:
            return 0.0
        disc = b * b - 4.0 * a * c
        if disc < 0.0:
            return 0.0
        sdisc = disc ** 0.5
        s1 = (-b - sdisc) / (2.0 * a)
        s2 = (-b + sdisc) / (2.0 * a)
        cand = [s for s in (s1, s2) if s > self.eps]
        return float(min(cand)) if cand else 0.0

    def _reflective_position(self, q: torch.Tensor, p: torch.Tensor,
                             step_size: float, mass_diag: torch.Tensor):
        q = self._project_to_ball(q)
        rem = float(step_size)
        t = 0.0
        v = p / mass_diag

        for _ in range(self.max_reflections):
            if t >= rem - 1e-15:
                break
            dt = rem - t
            q_trial = q + dt * v
            if q_trial.norm().item() <= self.R + 1e-12:
                q = q_trial
                t = rem
                break
            s_hit = self._hit_time_ball(q, v)
            if s_hit <= 0.0:
                q = self._project_to_ball(q_trial)
                n_hat = self._outward_normal(q)
                p = self._reflect_momentum(p, n_hat, mass_diag)
                t = rem
                break
            if s_hit > dt:
                s_hit = dt
            q = q + s_hit * v
            n_hat = self._outward_normal(q)
            p = self._reflect_momentum(p, n_hat, mass_diag)
            v = p / mass_diag
            t += s_hit

        return q, p


    def reflective_leapfrog(self, q0: torch.Tensor, p0: torch.Tensor,
                            step_size: float, n_steps: int,
                            mass_diag: torch.Tensor):
        q, p = q0.clone(), p0.clone()
        for _ in range(n_steps):
            p = p - 0.5 * step_size * self.grad_U_tilde(q)
            if self.use_reflection:
                q, p = self._reflective_position(q, p, step_size, mass_diag)
            else:
                q = q + step_size * (p / mass_diag)
            p = p - 0.5 * step_size * self.grad_U_tilde(q)
        return q, p

    def hamiltonian(self, q: torch.Tensor, p: torch.Tensor,
                    mass_diag: torch.Tensor) -> torch.Tensor:
        U = self.U_tilde(q)
        K = 0.5 * torch.sum(p * p / mass_diag)
        return U + K


    def hmc_step(self, q: torch.Tensor, mass_diag: torch.Tensor):
        d = q.numel()
        p0 = torch.randn(d, dtype=self.dtype, device=self.device) * torch.sqrt(mass_diag)
        H0 = self.hamiltonian(q, p0, mass_diag)
        q_new, p_new = self.reflective_leapfrog(q, p0, self.step_size, self.num_steps, mass_diag)
        p_new = -p_new
        H1 = self.hamiltonian(q_new, p_new, mass_diag)
        dH = (H1 - H0).detach()
        accept_prob = float(torch.exp(torch.clamp(-dH, max=0.0)).item())
        u = float(torch.rand(()).item())
        if u < accept_prob:
            return q_new.detach(), True, accept_prob, float(dH.item())
        else:
            return q.detach(), False, accept_prob, float(dH.item())


    def sample(
        self,
        d: int,
        q_init: Optional[torch.Tensor] = None,
        n_samples: int = 2000,
        burnin: int = 2000,
        thin: int = 1,
        kept_only: bool = True,
        verbose: bool = True,
        print_every: int = 500,
        max_total_steps: int = 100000,
    ) -> Dict[str, Any]:

        device, dtype = self.device, self.dtype

        if self.mass_diag is None or self.mass_diag.numel() != d:
            self.mass_diag = torch.ones(d, dtype=dtype, device=device)


        if q_init is None:

            q = self._find_feasible_init(d)
        else:
            q = q_init.clone().to(device, dtype)

        if self.use_reflection:
            q = self._project_to_ball(q)


        if burnin > 0:
            w1 = max(1, int(0.5 * burnin))
            w2 = max(w1 + 1, int(0.9 * burnin))
        else:
            w1 = w2 = 0

        adaptor = None
        if self.adapt_step_size and burnin > 0:
            adaptor = DualAveragingStepSize(
                init_step_size=self.step_size,
                target_accept=self.target_accept,
            )

        rv = RunningDiagVar(d, device=device, dtype=dtype)

        samples_kept = []
        dH_kept = []
        acc = 0
        it = 0
        kept_count = 0
        n_thinned_checks = 0
        use_thinning = self.use_reflection or self.use_soft_barrier


        for _ in range(int(burnin)):
            it += 1
            q, accepted, a_prob, dH = self.hmc_step(q, self.mass_diag)
            acc += int(accepted)

            if adaptor is not None:
                if self.adapt_mass and (it > w1) and (it <= w2):
                    rv.update(q.detach())
                if self.adapt_mass and (it == w2 + 1):
                    var = rv.var()
                    self.mass_diag = make_mass_diag_from_var(
                        var, jitter=self.mass_jitter,
                        min_m=self.mass_min, max_m=self.mass_max,
                    ).to(device=device, dtype=dtype)
                    adaptor = DualAveragingStepSize(
                        init_step_size=self.step_size,
                        target_accept=self.target_accept,
                    )
                new_eps = adaptor.update(a_prob)
                new_eps = max(self.eps_min, min(self.eps_max, new_eps))
                self.step_size = float(new_eps)

                if it == burnin:
                    final_eps = adaptor.final_step_size()
                    final_eps = max(self.eps_min, min(self.eps_max, final_eps))
                    self.step_size = float(final_eps)

            if verbose and (it % print_every == 0):
                print(f"[warmup {it}/{burnin}] acc={acc/it:.3f} eps={self.step_size:.4g}")

            if it >= max_total_steps:
                break


        if kept_only and use_thinning:
            while len(samples_kept) < n_samples and it < max_total_steps:
                it += 1
                q, accepted, a_prob, dH = self.hmc_step(q, self.mass_diag)
                acc += int(accepted)

                if (it - burnin) % thin != 0:
                    continue

                n_thinned_checks += 1
                if self.constraint.in_K(q):
                    samples_kept.append(q.detach().clone())
                    dH_kept.append(dH)
                    kept_count += 1

                if verbose and (it % print_every == 0):
                    print(f"[sample {it}] acc={acc/it:.3f} kept={len(samples_kept)}/{n_samples} eps={self.step_size:.4g}")
        else:
            target_total = n_samples * thin
            t_phase = 0
            for _ in range(target_total):
                if it >= max_total_steps:
                    break
                it += 1
                q, accepted, a_prob, dH = self.hmc_step(q, self.mass_diag)
                acc += int(accepted)
                t_phase += 1

                if t_phase % thin != 0:
                    continue

                n_thinned_checks += 1
                if use_thinning:
                    if self.constraint.in_K(q):
                        samples_kept.append(q.detach().clone())
                        dH_kept.append(dH)
                        kept_count += 1
                else:
                    samples_kept.append(q.detach().clone())
                    dH_kept.append(dH)
                    kept_count += 1

                if verbose and (it % print_every == 0):
                    print(f"[sample {it}] acc={acc/it:.3f} kept={len(samples_kept)} eps={self.step_size:.4g}")


        acc_rate = acc / max(it, 1)
        kept_fraction = kept_count / max(n_thinned_checks, 1)

        if len(samples_kept) == 0:
            chain = torch.empty(0, d, dtype=dtype, device=device)
        else:
            chain = torch.stack(samples_kept, dim=0)

        return {
            "chain": chain,
            "acc_rate": float(acc_rate),
            "kept_fraction": float(kept_fraction),
            "n_kept": len(samples_kept),
            "total_steps": int(it),
            "step_size": float(self.step_size),
            "mass_diag": self.mass_diag.detach().clone(),
        }

    def _find_feasible_init(self, d: int) -> torch.Tensor:
        """Try to find a feasible starting point inside K."""
        device, dtype = self.device, self.dtype

        if isinstance(self.constraint, AnnulusConstraint):

            mid_r = (self.constraint.a + self.constraint.b) / 2.0
            q = torch.zeros(d, dtype=dtype, device=device)
            q[0] = mid_r
            return q

        elif isinstance(self.constraint, SimplexConstraint):

            q = torch.full((d,), 1.0 / (d + 1), dtype=dtype, device=device)
            return q

        else:
            return torch.randn(d, dtype=dtype, device=device) * 0.1
