'''
useful functions
'''
import os
import torch
import optuna
import numpy as np 
from scipy.special import erf, erfinv


def best_model(study_name, output_dir):
    storage = 'sqlite:///%s/%s/%s.db' % (output_dir, study_name, study_name)
    study   = optuna.load_study(study_name=study_name, storage=storage)
    best    = study.best_trial
    fmodel  = os.path.join(output_dir, study_name, '%s.%i.pt' % (study_name, best.number))
    return torch.load(fmodel), best


def LF(Muv, z, phi):
    alpha, beta, gamma, Muv_s = phi
    return (10**(gamma * (z - 9.)))/(10**(0.4*(alpha + 1)*(Muv - Muv_s) + 10**(0.4*(beta + 1)*(Muv - Muv_s))))


def cdf_transform(x, bounds):
    """ Transform from a Gaussian (which is x) to a Uniform with bounds as input.
    """
    return _gaussian_cdf(x, 0, 1) * (bounds[1] - bounds[0]) + bounds[0]


def inv_cdf_transform(x, bounds):
    """ Transform from a Uniform with bounds (which is x) to a Gaussian
    """
    return _inv_gaussian_cdf((x - bounds[0]) / (bounds[1] - bounds[0]), 0, 1)

def _gaussian_cdf(x, mu, sigma):
    """ CDF of a Gaussian distribution.

    :math:`F(x) = \\frac{1}{2}(1 + erf(\\frac{x - \\mu}{\\sigma}))`
    """
    return 0.5 * (1 + erf((x - mu) / (np.sqrt(2) * sigma)))


def _inv_gaussian_cdf(x, mu, sigma):
    """ Inverse CDF of a Gaussian distribution.

    :math:`F^{-1}(x) = \\mu + \\sigma \\sqrt{2} \\text{erfinv}(2 \\times x - 1)`

    """
    return mu + sigma * np.sqrt(2) * erfinv(2 * x - 1)
