'''

Deep Set neural posterior estimator for population inference from sets of galaxies


A field is a set of K galaxies {x_k}, x_k = (z_obs, Muv_obs, sig_z, sig_Muv). The galaxies are exchangeable,
so the posterior p(omega | {x_k}) is permutation invariant. The estimator is

    phi   : per-galaxy MLP           x_k -> h_k
    pool  : masked mean (default)    {h_k} -> h          (sets may be padded to a common length with a mask)
    rho   : MLP                      h -> e
    flow  : conditional NSF (zuko)   q(omega | e)

trained by maximising log q(omega | {x_k}) over simulated (omega, field) pairs. Inputs and parameters are
standardised internally; `fit` sets the standardisation from the training data and the physical-unit methods
(`sample_posterior`, `posterior_log_prob`) undo it.

Mean pooling is exact in form for i.i.d. galaxies (the log-likelihood is a sum over galaxies) and was the best
pooling in the K = 1000 ablation once the training set was large enough
(experiments/2026-10-07_k1000_ablation); 'sum', 'max', 'meanmax' and attention pooling ('attn', a learned
query attending over the per-galaxy embeddings, concatenated with the mean) are available for comparison.


'''
import time
import numpy as np
import torch
import torch.nn as nn
import zuko


class DeepSetEncoder(nn.Module):
    ''' permutation-invariant set encoder: phi per element -> masked pooling -> rho

    :param d_obs: dimension of each set element
    :param d_phi: width of phi (and of rho's hidden layer)
    :param d_emb: dimension of the output summary
    :param pool: 'mean' | 'sum' | 'max' | 'meanmax' | 'attn'
    :param n_heads: attention heads for pool='attn'
    '''
    POOLS = ('mean', 'sum', 'max', 'meanmax', 'attn')

    def __init__(self, d_obs=4, d_phi=128, d_emb=64, pool='mean', n_heads=4):
        super().__init__()
        if pool not in self.POOLS:
            raise ValueError(f'pool must be one of {self.POOLS}')
        self.pool = pool
        self.phi = nn.Sequential(nn.Linear(d_obs, d_phi), nn.GELU(),
                                 nn.Linear(d_phi, d_phi), nn.GELU(),
                                 nn.Linear(d_phi, d_phi))
        d_pool = 2 * d_phi if pool in ('meanmax', 'attn') else d_phi
        if pool == 'attn':
            self.query = nn.Parameter(torch.randn(1, 1, d_phi) / d_phi ** 0.5)
            self.attn = nn.MultiheadAttention(d_phi, n_heads, batch_first=True)
        self.rho = nn.Sequential(nn.Linear(d_pool, d_phi), nn.GELU(), nn.Linear(d_phi, d_emb))

    def forward(self, x, mask=None):
        ''' x: (B, K, d_obs); mask: (B, K) bool, True for real elements (None: all real) -> (B, d_emb) '''
        h = self.phi(x)
        if mask is None:
            mask = torch.ones(x.shape[:2], dtype=torch.bool, device=x.device)
        m = mask.unsqueeze(-1)
        mean = (h * m).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
        if self.pool == 'mean':
            p = mean
        elif self.pool == 'sum':
            p = (h * m).sum(1)
        elif self.pool == 'max':
            p = h.masked_fill(~m, -1e9).max(1).values
        elif self.pool == 'meanmax':
            p = torch.cat([mean, h.masked_fill(~m, -1e9).max(1).values], dim=-1)
        else:
            q = self.query.expand(h.shape[0], -1, -1)
            a, _ = self.attn(q, h, h, key_padding_mask=~mask, need_weights=False)
            p = torch.cat([mean, a.squeeze(1)], dim=-1)
        return self.rho(p)


class DeepSetNPE(nn.Module):
    ''' q(omega | {x_k}): Deep Set encoder + conditional neural spline flow

    :param d_theta: number of population parameters
    :param d_obs: dimension of each galaxy's observation vector
    :param d_phi, d_emb, pool, n_heads: encoder settings (see DeepSetEncoder)
    :param n_transforms, n_bins, hidden: NSF settings (hidden: width of the two hidden layers)

    `log_prob` and `sample` work in standardised units (used for training); `posterior_log_prob` and
    `sample_posterior` take and return physical units.
    '''
    def __init__(self, d_theta=4, d_obs=4, d_phi=128, d_emb=64, n_transforms=4, n_bins=8, hidden=64,
                 pool='mean', n_heads=4):
        super().__init__()
        self.config = dict(d_theta=d_theta, d_obs=d_obs, d_phi=d_phi, d_emb=d_emb, n_transforms=n_transforms,
                           n_bins=n_bins, hidden=hidden, pool=pool, n_heads=n_heads)
        self.encoder = DeepSetEncoder(d_obs, d_phi, d_emb, pool=pool, n_heads=n_heads)
        self.flow = zuko.flows.NSF(features=d_theta, context=d_emb, bins=n_bins, transforms=n_transforms,
                                   hidden_features=[hidden, hidden])
        # standardisation (not part of the state dict, so checkpoints stay interchangeable; saved by `save`)
        for name, dim in [('obs_mean', d_obs), ('obs_std', d_obs), ('theta_mean', d_theta), ('theta_std', d_theta)]:
            self.register_buffer(name, torch.zeros(dim) if 'mean' in name else torch.ones(dim), persistent=False)

    # ------------------------------------------------------------------ standardised units
    def log_prob(self, theta, x, mask=None):
        ''' log q(theta | x) for standardised theta (B, d_theta), x (B, K, d_obs) '''
        return self.flow(self.encoder(x, mask)).log_prob(theta)

    def sample(self, x, n_samples=1000, mask=None):
        ''' samples from q(theta | x) for one standardised set x (K, d_obs) -> (n_samples, d_theta) '''
        emb = self.encoder(x.unsqueeze(0), None if mask is None else mask.unsqueeze(0))
        return self.flow(emb).sample((n_samples,)).squeeze(1)

    # ------------------------------------------------------------------ standardisation
    def set_normalization(self, obs_mean, obs_std, theta_mean, theta_std):
        for name, v in [('obs_mean', obs_mean), ('obs_std', obs_std), ('theta_mean', theta_mean), ('theta_std', theta_std)]:
            getattr(self, name).copy_(torch.as_tensor(np.asarray(v), dtype=torch.float32))

    def _x(self, x):
        return (torch.as_tensor(np.asarray(x), dtype=torch.float32, device=self.obs_mean.device) - self.obs_mean) / self.obs_std

    def _theta(self, theta):
        return (torch.as_tensor(np.asarray(theta), dtype=torch.float32, device=self.theta_mean.device) - self.theta_mean) / self.theta_std

    # ------------------------------------------------------------------ physical units
    @torch.no_grad()
    def sample_posterior(self, x, n_samples=1000, mask=None):
        ''' posterior samples in physical units for one field x (K, d_obs) -> (n_samples, d_theta) array '''
        self.eval()
        m = None if mask is None else torch.as_tensor(np.asarray(mask), dtype=torch.bool, device=self.obs_mean.device)
        s = self.sample(self._x(x), n_samples=n_samples, mask=m)
        return (s * self.theta_std + self.theta_mean).cpu().numpy()

    @torch.no_grad()
    def posterior_log_prob(self, theta, x, mask=None):
        ''' log q(theta | x) as a density in physical units; theta (B, d_theta), x (B, K, d_obs) -> (B,) array '''
        self.eval()
        m = None if mask is None else torch.as_tensor(np.asarray(mask), dtype=torch.bool, device=self.obs_mean.device)
        lp = self.log_prob(self._theta(theta), self._x(x), m)
        return (lp - torch.log(self.theta_std).sum()).cpu().numpy()

    # ------------------------------------------------------------------ I/O
    def save(self, path, **extra):
        ''' save weights, configuration and standardisation (plus any extra metadata) '''
        torch.save(dict(model_state=self.state_dict(), config=self.config,
                        obs_mean=self.obs_mean.cpu().numpy(), obs_std=self.obs_std.cpu().numpy(),
                        omega_mean=self.theta_mean.cpu().numpy(), omega_std=self.theta_std.cpu().numpy(), **extra), path)

    @staticmethod
    def config_from_state(state, pool=None, n_heads=4):
        ''' infer the architecture from a state dict (for checkpoints saved without a configuration).
        mean / sum / max pooling have identical weights, so `pool` is taken from the caller if given. '''
        d_phi, d_obs = state['encoder.phi.0.weight'].shape
        d_emb = state['encoder.rho.2.weight'].shape[0]
        if 'encoder.query' in state:
            pool = 'attn'
        elif state['encoder.rho.0.weight'].shape[1] == 2 * d_phi:
            pool = 'meanmax'
        elif pool is None or pool in ('meanmax', 'attn'):
            pool = 'mean'
        n_transforms = 1 + max(int(k.split('.')[3]) for k in state if k.startswith('flow.transform.transforms.'))
        hidden, d_in = state['flow.transform.transforms.0.hyper.0.weight'].shape
        d_theta = d_in - d_emb
        n_bins = (state['flow.transform.transforms.0.hyper.4.weight'].shape[0] // d_theta + 1) // 3
        return dict(d_theta=d_theta, d_obs=d_obs, d_phi=d_phi, d_emb=d_emb, n_transforms=n_transforms,
                    n_bins=n_bins, hidden=hidden, pool=pool, n_heads=n_heads)

    @classmethod
    def load(cls, path, map_location='cpu'):
        ''' load a checkpoint written by `save`, or by the experiment scripts (dicts with 'model_state' and the
        standardisation; the architecture is then inferred from the weights) '''
        ck = torch.load(path, map_location=map_location, weights_only=False)
        cfg = ck.get('config')
        if not cfg:
            old = ck.get('cfg') or ck.get('args') or {}
            cfg = cls.config_from_state(ck['model_state'], pool=old.get('pool'), n_heads=old.get('n_heads', 4))
        model = cls(**cfg)
        model.load_state_dict(ck['model_state'])
        model.set_normalization(ck['obs_mean'], ck['obs_std'], ck['omega_mean'], ck['omega_std'])
        model.eval()
        return model


def fit(model, theta, x, theta_val=None, x_val=None, mask=None, mask_val=None, val_fraction=0.05,
        lr=3e-4, batch_size=128, max_epochs=300, patience=30, clip=5., seed=0, verbose=0):
    ''' train a DeepSetNPE by maximum likelihood on simulated fields (physical units).

    Sets the model's standardisation from the training data, trains with Adam, a cosine learning-rate schedule
    and early stopping on the validation loss, and restores the best weights.

    :param theta: (N, d_theta) population parameters of the training fields
    :param x: (N, K, d_obs) galaxies of the training fields (padded to a common K if `mask` is given)
    :param theta_val, x_val, mask_val: validation fields; if None, `val_fraction` of the training fields
    :param mask: (N, K) bool, True for real galaxies
    :param verbose: print a progress line every `verbose` epochs (0: silent)
    :return: dict with train/validation loss curves, best validation loss and epoch, wall time
    '''
    theta, x = np.asarray(theta, np.float32), np.asarray(x, np.float32)
    mask = None if mask is None else np.asarray(mask, bool)
    rng = np.random.default_rng(seed)
    if theta_val is None:
        perm = rng.permutation(len(theta)); n_val = max(1, int(val_fraction * len(theta)))
        iv, it = perm[:n_val], perm[n_val:]
        theta_val, x_val = theta[iv], x[iv]; mask_val = None if mask is None else mask[iv]
        theta, x = theta[it], x[it]; mask = None if mask is None else mask[it]
    # standardisation from the training data (observations over real elements only, subsampled for speed)
    obs = x[mask] if mask is not None else x.reshape(-1, x.shape[-1])
    if len(obs) > 2_000_000:
        obs = obs[rng.choice(len(obs), 2_000_000, replace=False)]
    model.set_normalization(obs.mean(0), obs.std(0), theta.mean(0), theta.std(0))

    dev = model.obs_mean.device
    T = lambda a: None if a is None else torch.as_tensor(a, device=dev)
    th_tr, x_tr, m_tr = model._theta(theta), model._x(x), T(mask)
    th_va, x_va, m_va = model._theta(theta_val), model._x(x_val), T(None if mask_val is None else np.asarray(mask_val, bool))

    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_epochs)
    tr_losses, va_losses = [], []
    best_val, best_state, best_epoch, no_imp, t0 = np.inf, None, -1, 0, time.time()
    for epoch in range(max_epochs):
        model.train()
        perm = torch.randperm(len(th_tr), device=dev); bl = []
        for i in range(0, len(th_tr), batch_size):
            bi = perm[i:i + batch_size]
            loss = -model.log_prob(th_tr[bi], x_tr[bi], None if m_tr is None else m_tr[bi]).mean()
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt.step(); bl.append(loss.item())
        sched.step(); tr_losses.append(float(np.mean(bl)))
        model.eval()
        with torch.no_grad():   # mean NLL over validation fields, in chunks to bound memory
            vl = sum(-model.log_prob(th_va[i:i + 512], x_va[i:i + 512], None if m_va is None else m_va[i:i + 512]).sum().item()
                     for i in range(0, len(th_va), 512)) / len(th_va)
        va_losses.append(vl)
        if vl < best_val:
            best_val, best_epoch, no_imp = vl, epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            no_imp += 1
        if verbose and (epoch + 1) % verbose == 0:
            print(f'  epoch {epoch+1:4d}  train {tr_losses[-1]:.4f}  val {vl:.4f}  best {best_val:.4f} @ {best_epoch+1}'
                  f'  ({time.time()-t0:.0f}s)', flush=True)
        if no_imp >= patience:
            break
    model.load_state_dict(best_state)
    model.eval()
    return dict(train_losses=tr_losses, val_losses=va_losses, best_val=float(best_val), best_epoch=int(best_epoch) + 1,
                n_epochs=len(tr_losses), wall_time_s=float(time.time() - t0))
