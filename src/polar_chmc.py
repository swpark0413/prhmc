import time
from typing import Callable, Optional, Dict, Any, Tuple
import numpy as np
import torch
from src.utils import init_polar, DualAveragingStepSize, RunningDiagVar, make_mass_diag_from_var, align_filled_for_rhat, nanmax, nanmean, nanmin
import traceback

class ConstrainedPolarHMC:
    def __init__(
        self,
        logprob_fn: Callable[[torch.Tensor], torch.Tensor],
        step_size: float = 0.05,
        num_steps: int = 10,
        device: str = "cpu",
        dtype: torch.dtype = torch.float64,


        use_reflection: bool = True,
        use_soft_barrier: bool = False,


        beta_phiK: float = 10.0,
        lam_penalty: float = 10.0,
        rho: float = 0.1,


        barrier_pos: str = "hinge",
        barrier_sigma: float = 10.0,
        barrier_power: int = 4,
        barrier_delta: float = 1e-2,


        hit_time: bool = True,
        max_reflections: int = 50,
        eps: float = 1e-12,


        target_accept: float = 0.8,
        adapt_step_size: bool = True,
        eps_min: float = 1e-6,
        eps_max: float = 1.0,

        adapt_mass: bool = True,
        mass_jitter: float = 1e-3,
        mass_min: float = 1e-6,
        mass_max: float = 1e6,

    ):
        self.logprob_fn = logprob_fn
        self.step_size = float(step_size)
        self.num_steps = int(num_steps)
        self.device = device
        self.dtype = dtype

        self.use_reflection = bool(use_reflection)
        self.use_soft_barrier = bool(use_soft_barrier)

        self.beta_phiK = float(beta_phiK)
        self.lam = float(lam_penalty)
        self.rho = float(rho)

        self.barrier_sigma = float(barrier_sigma)
        self.barrier_power = int(barrier_power)

        self.barrier_delta = float(barrier_delta)
        self.barrier_pos = str(barrier_pos).lower()

        self.hit_time = bool(hit_time)
        self.max_reflections = int(max_reflections)
        self.eps = float(eps)


        self.target_accept = float(target_accept)
        self.adapt_step_size = bool(adapt_step_size)
        self.eps_min = float(eps_min)
        self.eps_max = float(eps_max)

        self.adapt_mass = bool(adapt_mass)
        self.mass_jitter = float(mass_jitter)
        self.mass_min = float(mass_min)
        self.mass_max = float(mass_max)


        self.r: Optional[int] = None
        self.u: Optional[int] = None
        self.c: Optional[float] = None
        self.M: Optional[float] = None
        self.R: Optional[float] = None


        self.mass_diag: Optional[torch.Tensor] = None


    def _set_geometry(self, r: int, u: int, c: float, M: float, R: float):
        self.r = int(r)
        self.u = int(u)
        self.c = float(c)
        self.M = float(M)
        self.R = float(R)

    def _check_geometry(self):
        if any(v is None for v in (self.r, self.u, self.c, self.M, self.R)):
            raise RuntimeError("Geometry not set. Call sample(..., r,u,c,M,R).")


    def violations_B(self, B: torch.Tensor, c: float, M: float):
        S = B.transpose(-1, -2) @ B
        eigvals = torch.linalg.eigvalsh(S)
        lam_min = eigvals[..., 0]
        v1 = c - lam_min
        fro2 = (B * B).sum(dim=(-2, -1))
        v2 = fro2 - M
        return v1, v2, lam_min, fro2

    def phi_K(self, B: torch.Tensor) -> torch.Tensor:
        self._check_geometry()
        v1, v2, _, _ = self.violations_B(B, c=self.c, M=self.M)
        V = torch.stack([v1, v2], dim=-1)
        return torch.logsumexp(self.beta_phiK * V, dim=-1) / self.beta_phiK


    def U_base(self, B: torch.Tensor) -> torch.Tensor:
        q = B.reshape(-1)
        return -self.logprob_fn(q)


    def phi_pos_softplus(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.softplus(self.barrier_sigma * x) / self.barrier_sigma


    def _smoothstep_quintic(self, s: torch.Tensor) -> torch.Tensor:
        return s**3 * (10 - 15*s + 6*s**2)


    def phi_pos_hinge(self, x: torch.Tensor) -> torch.Tensor:
        """
        C² smooth hinge (exactly 0 for x<=0)
        """
        delta = self.barrier_delta

        out = torch.zeros_like(x)

        neg = x <= 0
        pos = x >= delta
        mid = (~neg) & (~pos)

        out[pos] = x[pos]

        s = x[mid] / delta
        out[mid] = x[mid] * self._smoothstep_quintic(s)

        return out


    def phi_pos(self, x: torch.Tensor) -> torch.Tensor:
        mode = self.barrier_pos

        if mode == "softplus":
            return self.phi_pos_softplus(x)

        elif mode == "hinge":
            return self.phi_pos_hinge(x)

        elif mode == "none":
            return x

        else:
            raise ValueError(f"Unknown barrier_pos={mode}")

    def Psi_barrier(self, z: torch.Tensor) -> torch.Tensor:
        return z ** self.barrier_power

    def U_tilde_soft_barrier(self, q: torch.Tensor) -> torch.Tensor:
        self._check_geometry()
        B = q.view(self.r, self.u)
        U0 = self.U_base(B)

        phi = self.phi_K(B.unsqueeze(0))[0]

        z = self.phi_pos(phi) / self.rho
        return U0 + self.lam * self.Psi_barrier(z)

    def U_tilde(self, q: torch.Tensor) -> torch.Tensor:
        if self.use_soft_barrier:
            return self.U_tilde_soft_barrier(q)
        self._check_geometry()
        B = q.view(self.r, self.u)
        return self.U_base(B)

    def grad_U_tilde(self, q: torch.Tensor) -> torch.Tensor:
        q_req = q.clone().detach().requires_grad_(True)
        Uv = self.U_tilde(q_req)
        (gq,) = torch.autograd.grad(Uv, q_req, create_graph=False)
        return gq


    def phi_container(self, q: torch.Tensor) -> torch.Tensor:
        self._check_geometry()
        return (q * q).sum() - self.R ** 2

    def _project_to_ball(self, q: torch.Tensor) -> torch.Tensor:
        n = q.norm()
        if n.item() <= self.R:
            return q
        return q * (self.R / (n + self.eps))

    def _outward_unit_normal(self, q_on_boundary: torch.Tensor) -> torch.Tensor:
        return q_on_boundary / (q_on_boundary.norm() + self.eps)

    def _reflect_velocity(self, v: torch.Tensor, n_hat: torch.Tensor) -> torch.Tensor:

        return v - 2.0 * (v @ n_hat) * n_hat

    def _reflect_momentum_mass(self, p: torch.Tensor, n_hat: torch.Tensor, mass_diag: torch.Tensor) -> torch.Tensor:
        mass_inv = 1.0 / mass_diag
        v = mass_inv * p
        v_ref = self._reflect_velocity(v, n_hat)
        return mass_diag * v_ref

    def _hit_time_ball(self, q: torch.Tensor, v: torch.Tensor) -> float:

        a = (v @ v).item()
        b = (2.0 * (q @ v)).item()
        c = ((q @ q).item() - self.R * self.R)
        if a <= self.eps:
            return 0.0
        disc = b * b - 4.0 * a * c
        if disc < 0.0:
            return 0.0
        sdisc = disc ** 0.5
        s1 = (-b - sdisc) / (2.0 * a)
        s2 = (-b + sdisc) / (2.0 * a)
        cand = [s for s in (s1, s2) if s > 0]
        return float(min(cand)) if cand else 0.0

    def _reflective_position_hit_time(self, q: torch.Tensor, p: torch.Tensor, step_size: float, mass_diag: torch.Tensor):
        q = self._project_to_ball(q)
        rem = float(step_size)
        t = 0.0

        mass_inv = 1.0 / mass_diag
        v = mass_inv * p

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
                n_hat = self._outward_unit_normal(q)
                p = self._reflect_momentum_mass(p, n_hat, mass_diag)
                t = rem
                break

            if s_hit > dt:
                s_hit = dt

            q = q + s_hit * v
            n_hat = self._outward_unit_normal(q)
            p = self._reflect_momentum_mass(p, n_hat, mass_diag)

            v = (1.0 / mass_diag) * p
            t += s_hit

        return q, p

    def _reflective_position_projection(self, q: torch.Tensor, p: torch.Tensor, step_size: float, mass_diag: torch.Tensor):
        mass_inv = 1.0 / mass_diag
        v = mass_inv * p
        q_new = q + step_size * v

        if self.phi_container(q_new).item() <= 0.0:
            return q_new, p

        q_proj = self._project_to_ball(q_new)
        n_hat = self._outward_unit_normal(q_proj)
        p = self._reflect_momentum_mass(p, n_hat, mass_diag)
        return q_proj, p


    def reflective_leapfrog(self, q0: torch.Tensor, p0: torch.Tensor, step_size: float, n_steps: int, mass_diag: torch.Tensor):
        q = q0.clone()
        p = p0.clone()
        mass_inv = 1.0 / mass_diag

        for _ in range(n_steps):
            p = p - 0.5 * step_size * self.grad_U_tilde(q)

            if self.use_reflection:
                if self.hit_time:
                    q, p = self._reflective_position_hit_time(q, p, step_size, mass_diag)
                else:
                    q, p = self._reflective_position_projection(q, p, step_size, mass_diag)
            else:
                q = q + step_size * (mass_inv * p)

            p = p - 0.5 * step_size * self.grad_U_tilde(q)

        return q, p

    def hamiltonian(self, q: torch.Tensor, p: torch.Tensor, mass_diag: torch.Tensor) -> torch.Tensor:
        U = self.U_tilde(q)
        K = 0.5 * torch.sum(p * (p / mass_diag))
        return U + K

    def hmc_step(self, q_current: torch.Tensor, mass_diag: torch.Tensor):
        d = q_current.numel()
        q = q_current.clone().to(self.device, self.dtype)

        p0 = torch.randn(d, dtype=self.dtype, device=self.device) * torch.sqrt(mass_diag)
        H0 = self.hamiltonian(q, p0, mass_diag)

        q_new, p_new = self.reflective_leapfrog(q, p0, self.step_size, self.num_steps, mass_diag)
        p_new = -p_new
        H1 = self.hamiltonian(q_new, p_new, mass_diag)

        dH = (H1 - H0).detach()
        accept_prob = float(torch.exp(torch.clamp(-dH, max=0.0)).item())
        u = float(torch.rand((), device=self.device, dtype=self.dtype).item())
        accept = (u < accept_prob)

        if accept:
            return q_new.detach(), True, accept_prob, float(dH.item())
        else:
            return q_current.detach(), False, accept_prob, float(dH.item())


    def Gamma_from_B(self, B: torch.Tensor) -> torch.Tensor:
        U, _, Vh = torch.linalg.svd(B, full_matrices=False)
        return U @ Vh


    def sample_B_Gamma(
        self,
        q_init: Optional[torch.Tensor],
        *,
        R: float,
        r: int,
        u: int,
        c: float,
        M: float,
        n_samples: int,
        burnin: int,
        thin: int = 1,
        kept_only: bool = True,
        verbose: bool = True,
        print_every: int = 500,
        max_total_steps: int = 50000,
    ) -> Dict[str, Any]:

        status = "OK"
        error = None
        tb = None
        error_phase = None
        error_iter = None

        self._set_geometry(r=r, u=u, c=c, M=M, R=R)
        d = r * u
        device, dtype = self.device, self.dtype

        if (self.mass_diag is None) or (self.mass_diag.numel() != d):
            self.mass_diag = torch.ones(d, dtype=dtype, device=device)
        else:
            self.mass_diag = self.mass_diag.to(device=device, dtype=dtype)


        effective_kept_only = bool(kept_only) and (self.use_reflection or self.use_soft_barrier)
        run_until_kept = bool(effective_kept_only)
        if q_init is None:
            q = init_polar(r, u, eps=0.05, device=device, dtype=dtype).reshape(-1)
            q = self._project_to_ball(q) if self.use_reflection else q
        else:
            q = q_init.clone().view(-1).to(device, dtype)
            q = self._project_to_ball(q) if self.use_reflection else q


        if burnin > 0:
            w1 = max(1, int(0.5 * burnin))
            w2 = max(w1 + 1, int(0.9 * burnin))
        else:
            w1 = w2 = 0

        adaptor = None
        if self.adapt_step_size and burnin > 0:
            adaptor = DualAveragingStepSize(
                init_step_size=float(self.step_size),
                target_accept=float(self.target_accept),
            )

        rv = RunningDiagVar(d, device=device, dtype=dtype)

        B_kept, G_kept, H_kept = [], [], []
        acc = 0
        it = 0
        n_thinned_checks = 0
        kept_count = 0
        t_phase = 0
        dH = torch.tensor(0.0, device=device, dtype=dtype)

        try:
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
                            var,
                            jitter=self.mass_jitter,
                            min_m=self.mass_min,
                            max_m=self.mass_max,
                        ).to(device=device, dtype=dtype)

                        adaptor = DualAveragingStepSize(
                            init_step_size=float(self.step_size),
                            target_accept=float(self.target_accept),
                        )

                    new_eps = adaptor.update(a_prob)
                    new_eps = max(self.eps_min, min(self.eps_max, new_eps))
                    self.step_size = float(new_eps)

                    if it == burnin:
                        final_eps = adaptor.final_step_size()
                        final_eps = max(self.eps_min, min(self.eps_max, final_eps))
                        self.step_size = float(final_eps)

                if verbose and (it % print_every == 0 or it == burnin):
                    msg = (
                        f"[warmup iter {it}/{burnin}] "
                        f"acc.rate={acc/max(it,1):.3f}, eps={self.step_size:.4g}"
                    )
                    if self.adapt_mass:
                        msg += f", mass_adapt={'ON' if (it > w1) else 'OFF'}"
                    print(msg)

                if it >= max_total_steps:
                    break


            is_naive = (not self.use_reflection) and (not self.use_soft_barrier)
            use_K_thinning = (not is_naive)
            run_until_kept = bool(kept_only) and use_K_thinning


            if it < max_total_steps:
                if run_until_kept:
                    t_phase = 0
                    while (len(B_kept) < int(n_samples)) and (it < int(max_total_steps)):
                        it += 1
                        q, accepted, a_prob, dH = self.hmc_step(q, self.mass_diag)
                        acc += int(accepted)
                        t_phase += 1

                        if (t_phase % int(thin)) != 0:
                            if verbose and (it % print_every == 0):
                                print(f"[sample iter {it}] acc.rate={acc/max(it,1):.3f}, kept={len(B_kept)}, eps={self.step_size:.4g}")
                            continue

                        n_thinned_checks += 1
                        B = q.view(r, u)

                        phi = float(self.phi_K(B.unsqueeze(0))[0].item())
                        inK = (phi <= 0.0)
                        if inK:
                            G = self.Gamma_from_B(B.unsqueeze(0))[0]
                            B_kept.append(B.detach().clone())
                            G_kept.append(G.detach().clone())
                            H_kept.append(dH)
                            kept_count += 1

                        if verbose and (it % print_every == 0):
                            print(f"[sample iter {it}] acc.rate={acc/max(it,1):.3f}, kept={len(B_kept)}, eps={self.step_size:.4g}, phi={phi:.4g}, inK={int(inK)}")

                else:
                    target_total = int(n_samples) * int(thin)
                    t_phase = 0

                    for _ in range(target_total):
                        if it >= max_total_steps:
                            break
                        it += 1
                        q, accepted, a_prob, dH = self.hmc_step(q, self.mass_diag)
                        acc += int(accepted)
                        t_phase += 1

                        if (t_phase % int(thin)) != 0:
                            if verbose and (it % print_every == 0):
                                print(f"[sample iter {it}] acc.rate={acc/max(it,1):.3f}, kept={len(B_kept)}, eps={self.step_size:.4g}")
                            continue

                        n_thinned_checks += 1
                        B = q.view(r, u)

                        if not use_K_thinning:
                            G = self.Gamma_from_B(B.unsqueeze(0))[0]
                            B_kept.append(B.detach().clone())
                            G_kept.append(G.detach().clone())
                            H_kept.append(dH)
                            kept_count += 1
                        else:
                            phi = float(self.phi_K(B.unsqueeze(0))[0].item())
                            inK = (phi <= 0.0)
                            if inK:
                                G = self.Gamma_from_B(B.unsqueeze(0))[0]
                                B_kept.append(B.detach().clone())
                                G_kept.append(G.detach().clone())
                                H_kept.append(dH)
                                kept_count += 1

                        if verbose and (it % print_every == 0):
                            if use_K_thinning:
                                print(f"[sample iter {it}] acc.rate={acc/max(it,1):.3f}, kept={len(B_kept)}, eps={self.step_size:.4g}, phi={phi:.4g}")
                            else:
                                print(f"[sample iter {it}] acc.rate={acc/max(it,1):.3f}, kept={len(B_kept)}, eps={self.step_size:.4g}")

        except Exception as e:

            status = "ERROR"
            error = repr(e)
            tb = traceback.format_exc()

            error_phase = "warmup" if it <= int(burnin) else "sample"
            error_iter = int(it)

            if verbose:
                print(f"[{status}] phase={error_phase}, iter={error_iter}")
                print(error)

                print(tb)


        acc_rate = acc / max(it, 1)

        if len(B_kept) == 0:
            B_chain = torch.empty(0, r, u, device=device, dtype=dtype)
            Q_chain = torch.empty(0, r, u, device=device, dtype=dtype)
            dH_chain = torch.empty(0, device=device, dtype=dtype)
        else:
            B_chain = torch.stack(B_kept, dim=0)
            Q_chain = torch.stack(G_kept, dim=0)
            dH_chain = torch.tensor(H_kept, device=device, dtype=dtype)


        kept_rate = float(B_chain.shape[0] / max(int(n_samples), 1))


        kept_fraction = float(kept_count / max(n_thinned_checks, 1))

        return {
            "status": status,
            "error": error,
            "traceback": tb,
            "error_phase": error_phase,
            "error_iter": error_iter,

            "B_chain": B_chain,
            "Q_chain": Q_chain,
            "acc_rate": float(acc_rate),
            "kept_rate": float(kept_rate),
            "kept_fraction": float(kept_fraction),
            "step_size": float(self.step_size),
            "mass_diag": self.mass_diag.detach().clone(),
            "warmup_windows": (int(w1), int(w2), int(burnin)),
            "total_steps": int(it),
            "run_until_kept": bool(run_until_kept),
            "n_thinned_checks": int(n_thinned_checks),
            'dH': dH_chain
        }


def sample_fixed_kept(
    sampler: ConstrainedPolarHMC,
    *,
    q_init: Optional[torch.Tensor],
    R: float,
    r: int,
    u: int,
    c: float,
    M: float,
    n_samples: int,
    burnin: int,
    thin: int = 1,
    kept_only: bool = True,
    verbose: bool = True,
    print_every: int = 500,
    max_total_steps: int = 100000,
    chain_id: int = 0,
) -> Dict[str, Any]:
    if verbose:
        print(f"[chain {chain_id}] start | burnin={burnin}, n_samples={n_samples}, thin={thin}")

    t0 = time.perf_counter()
    out = sampler.sample_B_Gamma(
        q_init=q_init,
        R=R, r=r, u=u, c=c, M=M,
        n_samples=n_samples,
        burnin=burnin,
        thin=thin,
        kept_only=kept_only,
        verbose=verbose,
        print_every=print_every,
        max_total_steps=max_total_steps,
    )
    runtime = time.perf_counter() - t0

    out2 = dict(out)
    out2["runtime_sec"] = float(runtime)
    out2["run_until_kept"] = bool(out["run_until_kept"])
    out2["n_kept_actual"] = int(out["B_chain"].shape[0])

    if verbose:
        print(
            f"[chain {chain_id}] done | steps={out['total_steps']} | time={runtime:.2f}s | "
            f"acc={out['acc_rate']:.3f} | kept_rate={out['kept_rate']:.3f} | "
            f"eps={out['step_size']:.4g}"
        )
    return out2

def _empty_chain(device: str, dtype: torch.dtype, r: int, u: int):
    B = torch.empty(0, r, u, device=device, dtype=dtype)
    Q = torch.empty(0, r, u, device=device, dtype=dtype)
    dH = torch.empty(0, device=device, dtype=dtype)
    return B, Q, dH

def _pad_to_nkept(B: torch.Tensor, Q: torch.Tensor, dH: torch.Tensor, n_kept: int, r: int, u: int):
    """
    Pad (t, r, u) chains to (n_kept, r, u) with NaNs so torch.stack always works.
    """
    device, dtype = B.device, B.dtype
    t = int(B.shape[0])

    if t >= n_kept:
        return B[:n_kept], Q[:n_kept], dH[:n_kept]

    padB = torch.full((n_kept - t, r, u), float("nan"), device=device, dtype=dtype)
    padQ = torch.full((n_kept - t, r, u), float("nan"), device=device, dtype=dtype)
    paddH = torch.full((n_kept - t,), float("nan"), device=device, dtype=dtype)

    return torch.cat([B, padB], dim=0), torch.cat([Q, padQ], dim=0), torch.cat([dH, paddH], dim=0)


def run_chains(
    sampler_factory,
    *,
    R: float,
    r: int,
    u: int,
    c: float,
    M: float,
    n_kept: int,
    burnin: int,
    thin: int = 1,
    n_chains: int = 4,
    progress_every: int = 500,
    verbose: bool = True,
    kept_only: bool = False,
    max_total_steps_per_chain: int = 50000,
    seeds: Optional[Tuple[int, ...]] = None,
    device: Optional[str] = None,
    dtype: Optional[torch.dtype] = None,
    do_hungarian: bool = True,
    cond_eps: float = 1e-12,
):

    if seeds is None:
        seeds = tuple(range(1, n_chains + 1))
    if len(seeds) < n_chains:
        raise ValueError("seeds length must be >= n_chains")

    samplers = []
    for ch in range(n_chains):
        torch.manual_seed(int(seeds[ch]))
        np.random.seed(int(seeds[ch]))
        s = sampler_factory()
        if device is not None:
            s.device = device
        if dtype is not None:
            s.dtype = dtype
        samplers.append(s)


    dev0 = samplers[0].device
    dt0 = samplers[0].dtype


    Q_budget_list, B_budget_list, dH_budget_list, q_last_list = [], [], [], []

    budget_steps, budget_rt, budget_acc, budget_kept_rate, budget_n_kept = [], [], [], [], []
    budget_status, budget_error = [], []
    chain_ok = [True] * n_chains

    for ch in range(n_chains):
        seed = int(seeds[ch])
        torch.manual_seed(seed)
        np.random.seed(seed)

        if verbose:
            print("=" * 60)
            print(f"[BUDGET START] chain{ch}/{n_chains-1} seed={seed}")
            print("=" * 60)

        t0 = time.perf_counter()
        out = sample_fixed_kept(
            samplers[ch],
            q_init=None,
            R=R, r=r, u=u, c=c, M=M,
            n_samples=int(n_kept),
            burnin=int(burnin),
            thin=int(thin),
            kept_only=bool(kept_only),
            verbose=verbose,
            print_every=progress_every,
            max_total_steps=max_total_steps_per_chain,
            chain_id=ch,
        )
        rt = time.perf_counter() - t0
        st = out.get("status", "OK")
        budget_status.append(st)
        budget_error.append(out.get("error", None))

        if "B_chain" in out and torch.is_tensor(out["B_chain"]):
            B_budget = out["B_chain"].detach()
        else:
            B_budget, _, _ = _empty_chain(dev0, dt0, r, u)

        if "Q_chain" in out and torch.is_tensor(out["Q_chain"]):
            Q_budget = out["Q_chain"].detach()
        else:
            _, Q_budget, _ = _empty_chain(dev0, dt0, r, u)

        if "dH" in out:
            if torch.is_tensor(out["dH"]):
                dH_budget = out["dH"].detach().to(device=dev0, dtype=dt0)
            else:
                dH_budget = torch.tensor(out["dH"], device=dev0, dtype=dt0)
        else:
            _, _, dH_budget = _empty_chain(dev0, dt0, r, u)

        if st != "OK":
            chain_ok[ch] = False

        B_budget_list.append(B_budget)
        Q_budget_list.append(Q_budget)
        dH_budget_list.append(dH_budget)


        if B_budget.numel() > 0:
            q_last_list.append(B_budget[-1].reshape(-1).to(dev0, dt0))
        else:
            q_last_list.append(None)

        budget_steps.append(int(out["total_steps"]))
        budget_rt.append(float(rt))
        budget_acc.append(float(out["acc_rate"]))
        budget_kept_rate.append(float(out.get("kept_rate", np.nan)))
        budget_n_kept.append(int(out["Q_chain"].shape[0]))

        if verbose:
            print(f"[BUDGET DONE] chain{ch}: steps={out['total_steps']}, time={rt:.2f}s, kept={out['Q_chain'].shape[0]}")


    Q_filled, B_filled, dH_filled = [], [], []
    fill_steps, fill_rt = [], []
    fill_status, fill_error = [], []

    for ch in range(n_chains):
        if not chain_ok[ch]:

            if verbose:
                print("=" * 60)
                print(f"[FILL] chain{ch}: SKIP (budget status={budget_status[ch]})")
                print("=" * 60)

            Bcat = B_budget_list[ch]
            Qcat = Q_budget_list[ch]
            dHcat = dH_budget_list[ch]

            Bcat, Qcat, dHcat = _pad_to_nkept(Bcat, Qcat, dHcat, n_kept, r, u)

            B_filled.append(Bcat)
            Q_filled.append(Qcat)
            dH_filled.append(dHcat)

            fill_steps.append(0)
            fill_rt.append(0.0)
            fill_status.append("SKIP_DUE_TO_BUDGET_ERROR")
            fill_error.append(None)
            continue

        have = int(Q_budget_list[ch].shape[0])
        need = int(max(0, n_kept - have))

        if verbose:
            print("=" * 60)
            print(f"[FILL] chain{ch}: have={have}, need={need}")
            print("=" * 60)

        B_chunks = [B_budget_list[ch]]
        Q_chunks = [Q_budget_list[ch]]
        dH_chunks = [dH_budget_list[ch]]

        if need > 0:
            t0 = time.perf_counter()
            out2 = samplers[ch].sample_B_Gamma(
                q_init=q_last_list[ch],
                R=R, r=r, u=u, c=c, M=M,
                n_samples=int(need),
                burnin=0,
                thin=int(thin),
                kept_only=True,
                verbose=verbose,
                print_every=progress_every,
                max_total_steps=max_total_steps_per_chain,
            )
            rt2 = time.perf_counter() - t0

            st2 = out2.get("status", "OK")
            err2 = out2.get("error", None)
            fill_status.append(st2)
            fill_error.append(err2)


            if "B_chain" in out2 and torch.is_tensor(out2["B_chain"]):
                B_chunks.append(out2["B_chain"].detach())
            if "Q_chain" in out2 and torch.is_tensor(out2["Q_chain"]):
                Q_chunks.append(out2["Q_chain"].detach())
            if "dH" in out2:
                if torch.is_tensor(out2["dH"]):
                    dH_chunks.append(out2["dH"].detach().to(device=dev0, dtype=dt0))
                else:
                    dH_chunks.append(torch.tensor(out2["dH"], device=dev0, dtype=dt0))

            fill_steps.append(int(out2.get("total_steps", 0)))
            fill_rt.append(float(rt2))
        else:
            fill_steps.append(0)
            fill_rt.append(0.0)
            fill_status.append("SKIP(need=0)")
            fill_error.append(None)

        Bcat = torch.cat(B_chunks, dim=0)
        Qcat = torch.cat(Q_chunks, dim=0)
        dHcat = torch.cat(dH_chunks, dim=0)
        Bcat, Qcat, dHcat = _pad_to_nkept(Bcat, Qcat, dHcat, n_kept, r, u)


        B_filled.append(Bcat)
        Q_filled.append(Qcat)
        dH_filled.append(dHcat)


    B_post = torch.stack(B_filled, dim=0)
    Q_post = torch.stack(Q_filled, dim=0)
    dH_post = torch.stack(dH_filled, dim = 0)

    if do_hungarian:
        if torch.isnan(Q_post).any().item():
            if verbose:
                print("[ALIGN] skipped because NaNs exist (partial/empty chains).")
        else:
            Q_before = Q_post.clone()
            Q_post = align_filled_for_rhat(Q_post)
            if verbose:
                print("alignment max abs diff:", (Q_post - Q_before).abs().max().item())


    samples = {"B": B_post, "Q": Q_post, "dH": dH_post}

    with torch.no_grad():
        valid = ~torch.isnan(B_post).reshape(B_post.shape[0], B_post.shape[1], -1).any(dim=-1)

        S = B_post.transpose(-1, -2) @ B_post
        eigs = torch.linalg.eigvalsh(S)

        lam_min = eigs[..., 0]
        lam_max = eigs[..., -1]
        lam_sum = eigs.sum(dim=-1)

        cond = lam_max / torch.clamp(lam_min, min=float(cond_eps))

        def _masked_nan(x: torch.Tensor, mask: torch.Tensor):
            return torch.where(mask, x, torch.nan)

        lam_min_chain = nanmin(_masked_nan(lam_min, valid), dim=1)
        lam_max_chain = nanmax(_masked_nan(lam_max, valid), dim=1)
        lam_sum_chain = nanmean(_masked_nan(lam_sum, valid), dim=1)
        cond_chain = nanmax(_masked_nan(cond, valid), dim=1)


    meta = {
        "n_chains": int(n_chains),
        "burnin": int(burnin),
        "thin": int(thin),
        "n_kept": int(n_kept),
        "seeds": [int(s) for s in seeds[:n_chains]],

        "total_steps_per_chain": np.asarray(budget_steps, int) + np.asarray(fill_steps, int),
        "runtime_sec_per_chain": np.asarray(budget_rt, float) + np.asarray(fill_rt, float),
        "acc_rate_per_chain": np.asarray(budget_acc, float),
        "kept_rate_per_chain": np.ones(n_chains, dtype=float),


        "budget_status_per_chain": budget_status,
        "budget_error_per_chain": budget_error,
        "fill_status_per_chain": fill_status,
        "fill_error_per_chain": fill_error,
        "chain_ok_for_fill": np.asarray(chain_ok, dtype=bool),

        "BtB_eigvals": eigs.detach().cpu().numpy(),
        "BtB_lam_min": lam_min.detach().cpu().numpy(),
        "BtB_lam_max": lam_max.detach().cpu().numpy(),
        "BtB_lam_sum": lam_sum.detach().cpu().numpy(),
        "BtB_cond": cond.detach().cpu().numpy(),
        "BtB_cond_eps": float(cond_eps),


        "BtB_min_eig_chain": lam_min_chain.detach().cpu().numpy(),
        "BtB_max_eig_chain": lam_max_chain.detach().cpu().numpy(),
        "BtB_sum_eig_chain": lam_sum_chain.detach().cpu().numpy(),
        "BtB_cond_chain": cond_chain.detach().cpu().numpy(),
    }
    budget = {
        "Q_list": Q_budget_list,
        "B_list": B_budget_list,
        "total_steps_per_chain": np.asarray(budget_steps, int),
        "runtime_sec_per_chain": np.asarray(budget_rt, float),
        "acc_rate_per_chain": np.asarray(budget_acc, float),
        "kept_rate_per_chain": np.asarray(budget_kept_rate, float),
        "n_kept_actual_per_chain": np.asarray(budget_n_kept, int),
        "burnin": int(burnin),
        "thin": int(thin),
        "n_budget_checks": int(n_kept),

        "status_per_chain": budget_status,
        "error_per_chain": budget_error,
    }

    return samples, budget, meta
