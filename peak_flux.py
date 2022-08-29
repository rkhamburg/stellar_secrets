import numpy as np
import matplotlib.pyplot as plt
import random
import time

from source_distributions import Redshift, Luminosity
from astro_calc import peak_flux, detect_grbs
from plotting import *

def GRB_Simulation(x, z0, l0, l1=None, l2=None, l3=None, z1=None, z2=None,
    z3=None):
    """
    Description needed.
    """
    z            = x[0]   # redshifts
    l            = x[1]   # luminosities
    corr         = x[2]   # detector correction / kcorrection [10-1000 keV]
    corr_50_300  = x[3]   # detector correction / kcorrection [50-300 keV]
    func         = x[4]   # detector probability function
    dl_cm        = x[5]   # distance luminosities [cm]
    dV_dz_Gpc3   = x[6]   # comoving volume element as function of redshift
    merger_rate  = x[7]   # merger rates as function of redshift
    sim_num      = int(x[8])   # number of simulated GRBs
    ldist        = x[9]   # luminosity function
    zdist        = x[10]  # rate density function

    # Redshift distribution parameters
    z0 = 10 ** z0
    # Luminosity distribution parameters
    if l2 is not None:
        l2 = 10 ** l2

    # Sample from luminosity distributions
    lsample = get_luminosity_samples(l, sim_num, l0, l1, l2, name=ldist)
    # Sample from redshift distributions
    zsample, grb_yr, idx = get_redshift_samples(z, dV_dz_Gpc3, sim_num, z0, \
        z1=z1, z2=z2, z3=z3, merger_rate=merger_rate, name=zdist)

    # Get distance luminosities corresponding to sampled redshifts
    dl_sample = dl_cm[idx]

    # Get detector corrections corresponding to sampled redshifts
    corr = corr[idx]
    corr_50_300 = corr_50_300[idx]

    # Calculate peak photon flux [ph/cm^2/s]
    pf = peak_flux(lsample, zsample, dl_sample, corr)
    pf_50_300 = peak_flux(lsample, zsample, dl_sample, corr_50_300)

    # Detect GRBs
    det_pf, det_pf_50_300 = detect_grbs(pf, kind='gbm', pf_50_300=pf_50_300,
        func=func)

    # Get expected rate of detected GRBs
    model_counts = get_model_counts(det_pf, grb_yr, sim_num, type='peakflux')
    model_counts[model_counts==0] = 1e-30

    # Print info about detections
    verbose = False
    if verbose is not False:
        print_info(model_counts, grb_yr)
    ''' Plot intrinsic and detected distributions '''
    return model_counts

def print_info(detection_hist, grb_yr):
    fov, livetime = get_instr_info('gbm')
    rate_in_FOV = grb_yr * fov * livetime
    num_detections = np.sum(detection_hist)
    print ('all sky GRB rate:', grb_yr * 4*np.pi)
    print ('rate in GBM FOV:', rate_in_FOV)
    print ('number of detections:', num_detections)
    print ('GBM detection fraction:', num_detections / rate_in_FOV)
    return

def get_luminosity_samples(l, Ndraws, l0, l1=None, l2=None, name=None):
    """
    Obtain a random sample of luminosities from the luminosity
    distribution

    Parameters
    ----------
    l: array
        independent luminosities spanning desired range

    Returns
    ----------
    cl_sample: array
        randomly sampled luminosities
    """
    dist = Luminosity(l, l0, L1=l1, L2=l2, lmin=l[0], lmax=l[-1], name=name)
    pdf = dist.luminosity_pdf()
    pdf /= np.sum(pdf)
    draws = np.random.random(Ndraws)
    samples = l[np.searchsorted(np.cumsum(pdf), draws)]
    #plot_luminosity_samples(samples)
    return samples

def get_redshift_samples(z, dV, Ndraws, z0, z1=None, z2=None, z3=None, \
    merger_rate=None, name=None):
    """
    Obtain a random sample of redshifts from the redshift
    distribution

    Parameters
    ----------
    z: array
        independent redshifts spanning entire
        desired range
    dV: comoving volume element [Gpc^2]

    Returns
    ---------
    cz_sample: array
        randomly sampled redshifts from the intrinsic
        distribution
    local_rate: float
        the amplitude or local rate of the redshift
        distribution [Gpc^-3 yr^-1]
    """
    r = Redshift(z, z0, r1=z1, r2=z2, r3=z3, name=name)
    pdf, RGRB = r.source_rate_density(dV, merger_rate=merger_rate)
    draws = np.random.random(Ndraws)
    idx = np.searchsorted(np.cumsum(pdf), draws)
    #plot_redshift_samples(z[idx])
    return z[idx], RGRB, idx

def get_model_counts(dist, grb_yr, sim_num, type='peakflux', instrument='gbm'):
    d = np.histogram(dist, bins=get_bins(type))[0]
    d = d / np.sum(d)
    fov, livetime = get_instr_info(instrument)
    return grb_yr * fov * livetime * (len(dist)/sim_num) * d

def get_instr_info(d):
    if d == 'gbm':
        fov = 8. * np.pi / 3.
        livetime = 0.85
        return fov, livetime
