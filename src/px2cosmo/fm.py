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

z_grid  = np.linspace(4., 15., 1000)
dl_grid = Planck13.luminosity_distance(z_grid).to(u.pc).value
dl_spline = CubicSpline(z_grid, dl_grid)


def forwardmodel(phi, name='test0', phi_amp=6e-3): 
    ''' forward model noise model 
    '''
    # sample LF 
    mock = sampleLF(phi, phi_amp=phi_amp)

    # apply survey selection  
    select = selection_function(mock, name='multiple') 

    if name == 'test0': 
        # no noise model 
        return mock[select]

    elif name == 'test1': 
        # homoskedastic noise model sig_z = 0.2 and sig_Muv = 0.2
        mock[:,0] += 0.2 * np.random.normal(size=mock.shape[0])
        mock[:,1] += 0.2 * np.random.normal(size=mock.shape[0])
        return mock[select] 

    elif name == 'test2': 
        # apply simple Gaussian noise model with noise that sufficiently 
        # encompass noise levels of the CEERS sample 
        sig_photoz = np.random.uniform(0.05, 1, size=mock.shape[0])
        mock_photoz = mock[:,0] + sig_photoz * np.random.normal(size=mock.shape[0])
        
        sig_Muv = np.random.uniform(0.05, 0.5, size=mock.shape[0])
        mock_Muv = mock[:,1] + sig_Muv * np.random.normal(size=mock.shape[0])

        return np.vstack([mock_photoz, mock_Muv, sig_photoz, sig_Muv]).T[select]


def sampleLF(phi, phi_amp=6e-3): 
    ''' sample UV luminosity function 
    '''
    UVLFlist = np.array([phi_amp * U.LF(MUVcenters, z, phi) for z in zlist])

    lambdalist = UVLFlist * np.outer(DeltaVlist, MUVwidths) #avg for Poisson
    
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


def selection_function(mock, name='mock0'): 
    ''' impose selection function 
    '''
    if name == 'single': 
        prob_z = np.exp(-(mock[:,0] - 9.0)**2)
        select_z = prob_z > np.random.uniform(size=mock.shape[0])

        # convert mock absolute magnitude to apparent magnitudes
        dl = dl_spline(mock[:,0])
        mock_muv = mock[:,1] + 5 * np.log10(dl / 10.) 

        # m_uv selection weights
        w_muv_select = _select_muv(mock_muv, c_erf0=1., c_erf1=32)
        select_muv = np.random.uniform(size=mock.shape[0]) < w_muv_select

        return select_z & select_muv

    elif name == 'multiple': 
        prob_z0 = np.exp(-(mock[:,0] - 7.0)**2)
        select_z0 = prob_z0 > np.random.uniform(size=mock.shape[0])

        prob_z1 = np.exp(-(mock[:,0] - 9.0)**2)
        select_z1 = (prob_z1 > np.random.uniform(size=mock.shape[0])) & (~select_z0)

        prob_z2 = np.exp(-(mock[:,0] - 11.0)**2)
        select_z2 = (prob_z2 > np.random.uniform(size=mock.shape[0])) & (~select_z0) & (~select_z1)

        select_z = select_z0 | select_z1 | select_z2
        
        # convert mock absolute magnitude to apparent magnitudes
        dl = dl_spline(mock[:,0])
        mock_muv = mock[:,1] + 5 * np.log10(dl / 10.) 

        # m_uv selection weights
        w_muv_select = _select_muv(mock_muv, c_erf0=1., c_erf1=32)
        select_muv = np.random.uniform(size=mock.shape[0]) < w_muv_select

        return select_z & select_muv


def _select_muv(muv, c_erf0=0.9, c_erf1 = 32):
    # m_uv based selection 
    return 0.5*(1-erf(c_erf0*(muv - c_erf1)))
