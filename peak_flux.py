import numpy as np
import matplotlib.pyplot as plt
import random
import time

from source_distributions import Redshift, Luminosity
from astro_calc import peak_flux, detect_grbs
from plotting import *

def GRB_Simulation(x, z0, l0, l1=None, l2=None, l3=None, z1=None, z2=None,
    z3=None, ml0=None, ml1=None, ml2=None, mz0=None):
    """
    Description needed.
    """
    z             = x[0]   # redshifts
    l             = x[1]   # luminosities
    corr          = x[2]   # detector correction / kcorrection [10-1000 keV]
    corr_50_300   = x[3]   # detector correction / kcorrection [50-300 keV]
    mcorr         = x[4]   # detector correction / kcorrection [10-1000 keV]
    mcorr_50_300  = x[5]   # detector correction / kcorrection [50-300 keV]
    func          = x[6]   # detector probability function
    dl_cm         = x[7]   # distance luminosities [cm]
    dV_dz_Gpc3    = x[8]   # comoving volume element as function of redshift
    merger_rate   = x[9]   # merger rates as function of redshift
    sim_num       = int(x[10])   # number of simulated GRBs
    ldist         = x[11]  # luminosity function
    zdist         = x[12]  # rate density function
    mldist        = x[13]  # merger luminosity function

    # Redshift distribution parameters
    z0 = 10 ** z0
    if mz0 is not None:
        mz0 = 10 ** mz0
    # Luminosity distribution parameters
    if l2 is not None:
        l2 = 10 ** l2
    if ml2 is not None:
        ml2 = 10 ** ml2

    # Sample from luminosity distributions
    lsample      = get_luminosity_samples(l[0], sim_num, l0, l1, l2, name=ldist)
    lsample_merg = get_luminosity_samples(l[1], sim_num, ml0, ml1, ml2, \
        name=mldist)

    # Sample from redshift distributions
    zsample, grb_yr, idx        = get_redshift_samples(z, dV_dz_Gpc3, sim_num,
        z0, z1=z1, z2=z2, z3=z3, name=zdist)
    zsample_merg, mgrb_yr, midx = get_redshift_samples(z, dV_dz_Gpc3, sim_num,
        mz0, merger_rate=merger_rate, name='BNS')

    # Get distance luminosities corresponding to sampled redshifts
    dl_sample  = dl_cm[idx]
    mdl_sample = dl_cm[midx]

    # Get detector corrections corresponding to sampled redshifts
    corr         = corr[idx]
    mcorr        = mcorr[midx]
    corr_50_300  = corr_50_300[idx]
    mcorr_50_300 = mcorr_50_300[midx]

    # Calculate peak photon flux [ph/cm^2/s]
    pf             = peak_flux(lsample, dl_sample, corr)
    pf_50_300      = peak_flux(lsample, dl_sample, corr_50_300)
    pf_merg        = peak_flux(lsample_merg, mdl_sample, mcorr)
    pf_merg_50_300 = peak_flux(lsample_merg, mdl_sample, mcorr_50_300)
    #plot_peak_flux(pf, pf2=pf_merg)

    # Detect GRBs
    det_pf, _, _, det_lum = detect_grbs(pf, kind='gbm', pf_50_300=pf_50_300,
        func=func, z=zsample, L=lsample)
    det_pf_merg, _, _, det_lum_merg = detect_grbs(pf_merg, kind='gbm',
        pf_50_300=pf_merg_50_300, func=func, z=zsample_merg, L=lsample_merg)

    # Get expected rate of detected GRBs
    # Collapsars
    pf_counts  = get_model_counts(det_pf, grb_yr, sim_num, type='peakflux')
    lum_counts = get_model_counts(det_lum, grb_yr, sim_num, type='luminosity')
    pf_counts[pf_counts==0]   = 1e-30
    lum_counts[lum_counts==0] = 1e-30
    # Mergers
    pf_counts_merg  = get_model_counts(det_pf_merg, mgrb_yr, sim_num,
        type='peakflux')
    lum_counts_merg = get_model_counts(det_lum_merg, mgrb_yr, sim_num,
        type='merger luminosity')
    pf_counts_merg[pf_counts_merg==0]   = 1e-30
    lum_counts_merg[lum_counts_merg==0] = 1e-30
    plot_data(pf_counts, type='peakflux')
    plot_data(pf_counts_merg, type='peakflux')

    # Add distributions together
    all_pf_counts  = pf_counts  + pf_counts_merg
    all_lum_counts = lum_counts + lum_counts_merg
    print (np.sum(all_pf_counts), np.sum(pf_counts_merg))

    # Add conditional
    if np.sum(pf_counts_merg) < 0.05 * np.sum(all_pf_counts) \
        or np.sum(pf_counts_merg) > 0.5 * np.sum(all_pf_counts):
        print ('number of mergers not right')
        nan_arr = np.empty(len(get_bins('peakflux')))
        nan_arr[:] = np.nan
        return nan_arr

    # Print info about detections
    verbose = False
    if verbose is not False:
        print_info(all_pf_counts, grb_yr, grb_yr2=mgrb_yr)
    return np.array([all_pf_counts, all_lum_counts], dtype=object)


def print_info(detection_hist, grb_yr, grb_yr2=None):
    fov, livetime = get_instr_info('gbm')
    rate_in_FOV = (grb_yr+grb_yr2) * fov * livetime
    num_detections = np.sum(detection_hist)
    print ('all sky GRB rate:', (grb_yr+mgrb_yr) * 4.*np.pi)
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

def get_redshift_samples(z, dV, Ndraws, z0, z1=None, z2=None, z3=None,
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
    r = Redshift(z, z0, r1=z1, r2=z2, r3=z3, name=name, mrate=merger_rate)
    pdf, RGRB = r.source_rate_density(dV)
    draws = np.random.random(Ndraws)
    idx = np.searchsorted(np.cumsum(pdf), draws)
    #plot_redshift_samples(z[idx])
    return z[idx], RGRB, idx


def get_model_counts(dist, grb_yr, sim_num, type='peakflux', instrument='gbm'):
    d = np.histogram(dist, bins=get_bins(type))[0]
    print (d)
    print (np.sum(d))
    d = d / np.sum(d)

    if type == 'peakflux':
        fov, livetime = get_instr_info(instrument)
        return grb_yr * fov * livetime * (len(dist)/sim_num) * d
    elif type == 'luminosity':
        zrate = 104.
        return zrate * d
    elif type == 'merger luminosity':
        zrate = 7.
        return zrate * d

def get_instr_info(d):
    if d == 'gbm':
        fov = 8. * np.pi / 3.
        livetime = 0.85
        return fov, livetime
