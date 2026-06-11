'''

sample population posterior using emcee 


'''
import os, sys
import numpy as np 

import emcee

from px2cosmo import fm as FM 
from px2cosmo import util as UT

study_name = sys.argv[1]

# generate mock observations
phi_true = np.array([-1.65, -1.5, -0.2, -19.5]) # fit by eye to zeus21 output

X_obs = torch.tensor(FM.forwardmodel(phi_true, name='test0').astype(np.float32))

# load best q(X|phi)
p_X_phi = UT.best_model(study_name, '/Users/chang/data/px2cosmo/test0/nde') 


def log_prior(phi):
    alpha, beta, gamma, Muv_s = phi
    if -1.8 < alpha < -1.5 and -1.8 < beta < -1.2 and -0.4 < gamma < 0 and -22 < Muv_s < -18:
        return 0.
    return -np.inf


def log_posterior(phi):
    lp = log_prior(phi)
    if not np.isfinite(lp):
        return -np.inf

    _phi = torch.tensor(phi.astype(np.float32))
    
    # log likelihood
    logp = np.sum(p_X_phi.log_prob_batched(X_obs, _phi).cpu().numpy())
    return logp


# sample population posterior using emcee 
ndim, nwalkers = 4, 20
var = np.array([0.1, 0.1, 0.02, 0.1])
p0 = phi_true[None,:] + var[None,:]*np.random.randn(nwalkers, ndim)

sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior)
_ = sampler.run_mcmc(p0, 1000, progress=True)


fig, axes = plt.subplots(ndim, figsize=(10, 7), sharex=True)
samples = sampler.get_chain()
for i in range(ndim):
    ax = axes[i]
    ax.plot(samples[:, :, i], "k", alpha=0.3)
    ax.plot([0, samples.shape[0]], [phi_true[i], phi_true[i]], c='C1', ls='--')
    ax.set_xlim(0, len(samples))
    ax.set_ylabel([r'$\alpha$', r'$\beta$', r'$\gamma$', r'$M_{\rm UV}^*$'][i], fontsize=20)
    ax.yaxis.set_label_coords(-0.1, 0.5)
    ax.set_ylim([(-1.8, -1.5), (-1.8, -1.2), (-0.4, 0.0), (-22, -18)][i])

fig = DFM.corner(sampler.flatchain[5000:,:])
DFM.overplot_points(fig, [phi_true], color='red')
DFM.overplot_lines(fig, phi_true, color='red')
#DFM.overplot_lines(fig, np.median(sampler.flatchain[5000:,:], axis=0), color='blue')
