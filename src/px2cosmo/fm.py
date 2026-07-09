'''


forward model 



'''
import os
import numpy as np
from scipy.special import erf
from scipy.interpolate import CubicSpline

from astropy import units as u
from astropy.cosmology import Planck13

import zeus21

from . import util as U

# precompute the following for sampleLF
CosmoParams_input = zeus21.Cosmo_Parameters_Input(zmin_CLASS=0.0)
User_Parameters = zeus21.User_Parameters()  # add this

CosmoParams, _, _, _ = zeus21.cosmo_wrapper(User_Parameters, CosmoParams_input)

Nz = 133
NMUV = 300

zlist   = np.linspace(4, 15, Nz)
dzuvlf  = np.diff(zlist)[0] #just for UVLF calculation

MUVcenters  = np.linspace(-15.,-24.,NMUV) #centers of bins
MUVwidths   = -np.diff(MUVcenters)
MUVwidths   = np.append(MUVwidths,MUVwidths[-1])
dMUV        = MUVwidths[0]

OmegaSurvey = 38.0*(1./60.)**2 * (np.pi/180)**2 #rad^2
#OmegaSurvey = 0.5 * (np.pi/180)**2 #rad^2
DeltaVlist = OmegaSurvey * CosmoParams.chiofzint(zlist)**2 * dzuvlf/CosmoParams.Hofzint(zlist)
DeltaVwidths = np.outer(DeltaVlist, MUVwidths)  # constant; precomputed once

z_grid  = np.linspace(4., 15., 1000)
dl_grid = Planck13.luminosity_distance(z_grid).to(u.pc).value
dl_spline = CubicSpline(z_grid, dl_grid)

_Z_CENTERS = {'z7': 7.0, 'z9': 9.0, 'z11': 11.0, 'z14': 14.0}


def forwardmodel(phi, name='test0', phi_amp=6e-3): 
    ''' forward model noise model 
    '''
    # sample LF 
    mock = sampleLF(phi, phi_amp=phi_amp)

    if name == 'test0': 
        # apply survey selection  
        select = selection_function(mock, name='multiple') 

        # no noise model 
        return mock[select]

    elif name == 'test1': 
        # apply survey selection  
        select = selection_function(mock, name='multiple') 

        # homoskedastic noise model sig_z = 0.2 and sig_Muv = 0.2
        mock[:,0] += 0.2 * np.random.normal(size=mock.shape[0])
        mock[:,1] += 0.2 * np.random.normal(size=mock.shape[0])
        return mock[select] 

    elif name == 'test2': 
        # apply survey selection  
        select = selection_function(mock, name='multiple') 

        # apply simple Gaussian noise model with noise that sufficiently 
        # encompass noise levels of the CEERS sample 
        sig_photoz = np.random.uniform(0.05, 1, size=mock.shape[0])
        mock_photoz = mock[:,0] + sig_photoz * np.random.normal(size=mock.shape[0])
        
        sig_Muv = np.random.uniform(0.05, 0.5, size=mock.shape[0])
        mock_Muv = mock[:,1] + sig_Muv * np.random.normal(size=mock.shape[0])

        return np.vstack([mock_photoz, mock_Muv, sig_photoz, sig_Muv]).T[select]
    
    elif name in ['z7', 'z9', 'z11', 'z14']: 
        # apply survey selection  
        select = selection_function(mock, name=name) 

        # apply simple Gaussian noise model with noise that sufficiently 
        # encompass noise levels of the CEERS sample 
        sig_photoz = np.random.uniform(0.03, 1, size=mock.shape[0])
        mock_photoz = mock[:,0] + sig_photoz * np.random.normal(size=mock.shape[0])
        
        sig_Muv = np.random.uniform(0.01, 0.5, size=mock.shape[0])
        mock_Muv = mock[:,1] + sig_Muv * np.random.normal(size=mock.shape[0])

        return np.vstack([mock_photoz, mock_Muv, sig_photoz, sig_Muv]).T[select]


def sampleLF(phi, phi_amp=6e-3):
    ''' sample UV luminosity function
    '''
    alpha, beta, gamma, Muv_s = phi
    # LF(Muv, z) = 10^(gamma*(z-9)) * f(Muv) — factor into outer product
    z_factor = phi_amp * 10**(gamma * (zlist - 9.))                                         
    dmuv = MUVcenters - Muv_s
    inner = np.clip(0.4*(beta+1)*dmuv, -300, 300)
    outer = np.clip(0.4*(alpha+1)*dmuv + 10**inner, -300, 300)
    muv_factor = 1.0 / 10**outer
    UVLFlist = np.outer(z_factor, muv_factor)                                               

    lambdalist = UVLFlist * DeltaVwidths  #avg for Poisson
    
    Nsample = np.random.poisson(lambdalist, ((Nz, NMUV)))

    zz_grid, MUV_grid = np.meshgrid(zlist, MUVcenters, indexing='ij')
    counts = Nsample.ravel()
    mock = np.repeat(
        np.column_stack([zz_grid.ravel(), MUV_grid.ravel()]),
        counts,
        axis=0)

    # remove discreteness 
    mock[:,0] += dzuvlf * np.random.uniform(size=mock.shape[0])
    mock[:,1] += dMUV * np.random.uniform(size=mock.shape[0])
    return mock 


def selection_function(mock, name='z9'):
    ''' impose selection function
    '''
    z_c = _Z_CENTERS[name]
    prob_z = np.exp(-(mock[:,0] - z_c)**2)
    select_z = prob_z > np.random.uniform(size=mock.shape[0])

    dl = dl_spline(mock[:,0])
    mock_muv = mock[:,1] + 5 * np.log10(dl / 10.)
    w_muv_select = _select_muv(mock_muv, c_erf0=1., c_erf1=32)
    select_muv = np.random.uniform(size=mock.shape[0]) < w_muv_select

    return select_z & select_muv


def apply_selection_noise(mock, name):
    ''' Apply selection function and heteroskedastic noise to a sampled catalog.

    Returns array of shape (N, 4): [z_obs, Muv_obs, sig_z, sig_Muv]
    '''
    select = selection_function(mock, name=name)
    m = mock[select]
    sig_photoz = np.random.uniform(0.03, 1, size=m.shape[0])
    mock_photoz = m[:,0] + sig_photoz * np.random.normal(size=m.shape[0])
    sig_Muv = np.random.uniform(0.01, 0.5, size=m.shape[0])
    mock_Muv = m[:,1] + sig_Muv * np.random.normal(size=m.shape[0])
    return np.column_stack([mock_photoz, mock_Muv, sig_photoz, sig_Muv])


def _select_muv(muv, c_erf0=0.9, c_erf1 = 32):
    # m_uv based selection 
    return 0.5*(1-erf(c_erf0*(muv - c_erf1)))
