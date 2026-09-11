'''
Neural Score Estimator (NSE) for simulation-based inference via diffusion models.

Implements the VP-SDE with linear noise schedule from:
  Linhart et al. (2024), arXiv:2404.07593

Tall-data factorized scores:
  Geffner (simple): Geffner et al. (2023), arXiv:2209.14249
  GAUSS / JAC:      Linhart et al. (2024), arXiv:2404.07593
'''
import math

import torch
import torch.nn as nn
from torch import Tensor
from zuko.nn import MLP
from zuko.utils import broadcast


class NSE(nn.Module):
    '''Score network for the VP-SDE with linear noise schedule beta(t) = 32t.

    Learns eps(theta, x, t) such that score = -eps / sigma(t).
    At inference, combines single-observation scores for the tall-data setting
    using Geffner (simple), GAUSS, or JAC factorized scores.
    '''
    def __init__(self, theta_dim: int, x_dim: int, freqs: int = 3,
                 hidden_features: list = [128, 128]):
        super().__init__()
        self.theta_dim = theta_dim
        self.net = MLP(theta_dim + x_dim + 2 * freqs, theta_dim,
                       hidden_features=hidden_features, activation=nn.SiLU)
        self.register_buffer('freqs', torch.arange(1, freqs + 1) * math.pi)
        self.register_buffer('zeros', torch.zeros(theta_dim))
        self.register_buffer('ones',  torch.ones(theta_dim))

    def forward(self, theta: Tensor, x: Tensor, t: Tensor) -> Tensor:
        if t.dim() == 0:
            t = t.unsqueeze(0)
        t_enc = self.freqs * t[..., None]
        t_enc = torch.cat((t_enc.cos(), t_enc.sin()), dim=-1)
        theta, x, t_enc = broadcast(theta, x, t_enc, ignore=1)
        return self.net(torch.cat((theta, x, t_enc), dim=-1))

    def score(self, theta: Tensor, x: Tensor, t: Tensor) -> Tensor:
        return -self(theta, x, t) / self.sigma(t)

    # ── VP-SDE schedule (linear beta: beta(t) = 32t) ─────────────────────────

    def alpha(self, t: Tensor) -> Tensor:
        '''Mean scaling of the transition kernel: exp(-16 t^2).'''
        return torch.exp(-16 * t**2)

    def sigma(self, t: Tensor) -> Tensor:
        '''Std of the transition kernel: sqrt(1 - alpha(t) + eps).'''
        return torch.sqrt(1 - self.alpha(t) + math.exp(-16))

    # ── Samplers ──────────────────────────────────────────────────────────────

    @torch.no_grad()
    def ddim(self, shape, x: Tensor, steps: int = 500, eta: float = 1.0,
             prior_score_fn=None, mode: str = 'geffner',
             dist_cov_est: Tensor = None) -> Tensor:
        '''DDIM sampler for single or multiple observations.

        For n_obs=1 uses the standard (single-obs) score.
        For n_obs>1 uses the factorized tall-data score selected by `mode`.

        Args:
            shape: (n_samples,) — number of posterior draws.
            x: single observation (x_dim,) or stack (n_obs, x_dim).
            steps: DDIM discretisation steps.
            eta: stochasticity; eta=1 is DDPM, eta=0 is deterministic.
            prior_score_fn: callable (theta, t) -> score of diffused prior.
                Required when n_obs > 1.
            mode: 'geffner' | 'gauss' | 'jac'. Ignored when n_obs=1.
            dist_cov_est: per-observation posterior covariance estimate,
                shape (n_obs, theta_dim, theta_dim). Required for mode='gauss'.
                Obtain via estimate_posterior_covs().
        '''
        n_obs = x.shape[0] if x.dim() > 1 else 1
        use_tall = (n_obs > 1 and n_obs != shape[0])

        time = torch.linspace(1, 0, steps + 1).to(x)
        dt   = 1 / steps
        theta = torch.randn(shape[0], self.theta_dim, device=x.device, dtype=x.dtype)

        for t in time[:-1]:
            if not use_tall:
                score = self.score(theta, x, t).detach()
            elif mode == 'geffner':
                score = self._geffner_score(theta, x, t, prior_score_fn)
            elif mode == 'gauss':
                score = self._gauss_score(theta, x, t, prior_score_fn, dist_cov_est)
            elif mode == 'jac':
                score = self._jac_score(theta, x, t, prior_score_fn)
            else:
                raise ValueError(f"Unknown mode '{mode}'. Use 'geffner', 'gauss', or 'jac'.")

            alpha_t   = self.alpha(t)
            alpha_t_1 = self.alpha(t - dt)
            bridge_var = ((1 - alpha_t_1) / (1 - alpha_t)) * (1 - alpha_t / alpha_t_1)
            bridge_std = eta * bridge_var.clamp(min=0).sqrt()

            pred_theta_0 = alpha_t**(-0.5) * (theta + (1 - alpha_t) * score)
            est_noise    = (theta - alpha_t**0.5 * pred_theta_0) / (1 - alpha_t)**0.5
            theta_mean   = (alpha_t_1**0.5 * pred_theta_0
                            + (1 - alpha_t_1 - bridge_std**2).clamp(min=0).sqrt() * est_noise)
            theta = theta_mean + torch.randn_like(theta_mean) * bridge_std

        return theta

    @torch.no_grad()
    def ddim_batched(self, n_samples: int, x: Tensor, steps: int = 300,
                     eta: float = 1.0) -> Tensor:
        '''Batched single-observation DDIM for efficient TARP/MIRA validation.

        Runs n_samples DDIM chains for each observation in the batch
        simultaneously using tensor replication.

        Args:
            n_samples: posterior draws per observation.
            x: (n_test, x_dim) — one observation per test case.
            steps: DDIM steps.
        Returns:
            (n_test, n_samples, theta_dim)
        '''
        n_test = x.shape[0]
        x_rep  = x.unsqueeze(1).expand(-1, n_samples, -1).reshape(n_test * n_samples, -1)
        theta  = torch.randn(n_test * n_samples, self.theta_dim,
                              device=x.device, dtype=x.dtype)

        time = torch.linspace(1, 0, steps + 1).to(x)
        dt   = 1 / steps

        for t in time[:-1]:
            score = self.score(theta, x_rep, t).detach()

            alpha_t   = self.alpha(t)
            alpha_t_1 = self.alpha(t - dt)
            bridge_var = ((1 - alpha_t_1) / (1 - alpha_t)) * (1 - alpha_t / alpha_t_1)
            bridge_std = eta * bridge_var.clamp(min=0).sqrt()

            pred_theta_0 = alpha_t**(-0.5) * (theta + (1 - alpha_t) * score)
            est_noise    = (theta - alpha_t**0.5 * pred_theta_0) / (1 - alpha_t)**0.5
            theta_mean   = (alpha_t_1**0.5 * pred_theta_0
                            + (1 - alpha_t_1 - bridge_std**2).clamp(min=0).sqrt() * est_noise)
            theta = theta_mean + torch.randn_like(theta_mean) * bridge_std

        return theta.reshape(n_test, n_samples, self.theta_dim)

    @torch.no_grad()
    def estimate_posterior_covs(self, x: Tensor, n_samples: int = 500,
                                 steps: int = 100, eta: float = 0.5) -> Tensor:
        '''Estimate per-observation posterior covariance for GAUSS score.

        Runs DDIM independently for each x_i and computes the empirical
        posterior covariance. Required before calling ddim(..., mode='gauss').

        Args:
            x: (n_obs, x_dim)
            n_samples: DDIM samples per observation for covariance estimation.
            steps: DDIM steps for estimation (100 is usually sufficient).
        Returns:
            (n_obs, theta_dim, theta_dim)
        '''
        covs = []
        for i in range(x.shape[0]):
            samples = self.ddim((n_samples,), x=x[i:i+1], steps=steps, eta=eta)
            covs.append(torch.cov(samples.T))
        return torch.stack(covs)

    # ── Factorized scores ─────────────────────────────────────────────────────

    def _geffner_score(self, theta: Tensor, x: Tensor, t: Tensor,
                        prior_score_fn) -> Tensor:
        '''Geffner et al. factorized score (simple additive combination):
            score(theta | x_1,...,x_n) ≈ (1-n)*score_prior + sum_i score(theta | x_i)
        '''
        n_obs  = x.shape[0]
        scores = self.score(theta[:, None], x[None, :], t).detach()  # (N, n_obs, theta_dim)
        return (1 - n_obs) * prior_score_fn(theta, t) + scores.sum(dim=1)

    def _gauss_score(self, theta: Tensor, x: Tensor, t: Tensor,
                      prior_score_fn, dist_cov_est: Tensor) -> Tensor:
        '''GAUSS factorized score (Linhart et al. 2024).

        Uses pre-estimated per-observation posterior covariances as the
        backward kernel precision, yielding a precision-weighted combination.

        Args:
            dist_cov_est: (n_obs, theta_dim, theta_dim) from estimate_posterior_covs.
        '''
        n_obs, n_samples = x.shape[0], theta.shape[0]
        alpha_t = self.alpha(t)
        sigma_t = self.sigma(t)
        eye = torch.eye(self.theta_dim, device=t.device, dtype=theta.dtype)

        scores = self.score(theta[:, None], x[None, :], t).detach()  # (N, n_obs, theta_dim)

        # per-obs backward kernel precision: (n_obs, theta_dim, theta_dim)
        # Guard against non-finite covariances (collapsed ddim_batched samples near box
        # boundaries): replace bad rows with a unit-covariance fallback so they contribute
        # no more than the prior, then add a small diagonal regulariser before inverting.
        cov = dist_cov_est.to(theta.device, theta.dtype)
        bad = ~cov.isfinite().all(dim=-1).all(dim=-1)   # (n_obs,)
        if bad.any():
            cov = cov.clone()
            cov[bad] = eye
        cov = cov + 1e-4 * eye                          # diagonal regularisation
        prec_per_obs = torch.linalg.inv(cov) + (alpha_t / sigma_t**2) * eye
        # expand to (N, n_obs, theta_dim, theta_dim)
        prec_0_t = prec_per_obs.unsqueeze(0).expand(n_samples, -1, -1, -1)

        # prior precision for N(0,I): (1 + alpha_t/sigma_t^2) * I → (N, theta_dim, theta_dim)
        prec_prior = ((1 + alpha_t / sigma_t**2) * eye).unsqueeze(0).expand(n_samples, -1, -1)
        prior_score = prior_score_fn(theta, t)                             # (N, theta_dim)

        prec_score_prior = (prec_prior @ prior_score[..., None])[..., 0]  # (N, theta_dim)
        prec_score_post  = (prec_0_t @ scores[..., None])[..., 0]         # (N, n_obs, theta_dim)

        lda      = prec_prior * (1 - n_obs) + prec_0_t.sum(dim=1)         # (N, theta_dim, theta_dim)
        weighted = prec_score_prior + (prec_score_post - prec_score_prior[:, None]).sum(dim=1)

        return torch.linalg.solve(lda, weighted).detach()

    def _jac_score(self, theta: Tensor, x: Tensor, t: Tensor,
                    prior_score_fn) -> Tensor:
        '''JAC factorized score (Linhart et al. 2024).

        Computes per-sample, per-observation Jacobians of the score network
        via torch.func to estimate the backward kernel precision.
        More accurate than GAUSS but slower (O(N * n_obs) Jacobian evaluations).
        '''
        from torch.func import jacrev, vmap
        n_obs, n_samples = x.shape[0], theta.shape[0]
        alpha_t = self.alpha(t)
        sigma_t = self.sigma(t)
        eye = torch.eye(self.theta_dim, device=t.device, dtype=theta.dtype)

        def score_fn_aux(theta_i, x_j):
            s = self.score(theta_i[None], x_j[None], t)[0]
            return s, s

        with torch.enable_grad():
            jac_score, scores = vmap(
                lambda th: vmap(jacrev(score_fn_aux, has_aux=True), in_dims=(None, 0))(th, x)
            )(theta)
        # jac_score: (N, n_obs, theta_dim, theta_dim)
        # scores:    (N, n_obs, theta_dim)

        # backward kernel precision: (alpha_t/sigma_t^2) * (I + sigma_t^2 * J)
        prec_0_t = (alpha_t / sigma_t**2) * (eye + sigma_t**2 * jac_score)

        prec_prior  = ((1 + alpha_t / sigma_t**2) * eye).unsqueeze(0).expand(n_samples, -1, -1)
        prior_score = prior_score_fn(theta, t)

        prec_score_prior = (prec_prior @ prior_score[..., None])[..., 0]
        prec_score_post  = (prec_0_t @ scores[..., None])[..., 0]

        lda      = prec_prior * (1 - n_obs) + prec_0_t.sum(dim=1)
        weighted = prec_score_prior + (prec_score_post - prec_score_prior[:, None]).sum(dim=1)

        return torch.linalg.solve(lda, weighted).detach()


class NSELoss(nn.Module):
    '''Denoising score matching loss (noise parametrization).

    Given (theta, x) pairs, minimizes E[||eps(theta_t, x, t) - eps||^2]
    where theta_t = alpha(t)^0.5 * theta + sigma(t) * eps, eps ~ N(0,I).
    '''
    def __init__(self, estimator: NSE):
        super().__init__()
        self.estimator = estimator

    def forward(self, theta: Tensor, x: Tensor) -> Tensor:
        t       = torch.rand(theta.shape[0], dtype=theta.dtype, device=theta.device)
        scaling = self.estimator.alpha(t)**0.5
        sigma   = self.estimator.sigma(t)
        eps     = torch.randn_like(theta)
        theta_t = scaling[:, None] * theta + sigma[:, None] * eps
        return (self.estimator(theta_t, x, t) - eps).square().mean()


def gaussian_prior_score(theta_t: Tensor, t: Tensor, nse: NSE) -> Tensor:
    '''Score of the VP-diffused standard Gaussian prior N(0,I).

    For p_0 = N(0,I), the diffused marginal is N(0, (alpha(t)+sigma^2(t)) I).
    With the NSE schedule, alpha(t)+sigma^2(t) = 1+eps ≈ 1, so score ≈ -theta_t.
    '''
    denom = nse.alpha(t) + nse.sigma(t)**2
    return -theta_t / denom
