from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
from numpy.random import default_rng
rng = default_rng()

cosmo = FlatLambdaCDM(H0=70, Om0=0.3, Tcmb0=2.725)

def distance_luminosity(z):
    dl_Mpc = cosmo.luminosity_distance(z)
    dl_cm = dl_Mpc.to(u.cm)
    return dl_cm.value

def diff_comoving_volume(z):
    # Differential comoving volume dV [Gpc^3 / sr / dz]
    dV_dzdOm = cosmo.differential_comoving_volume(z)
    dV_dz_Gpc3 = dV_dzdOm.to(u.Gpc**3 / u.sr)
    return dV_dz_Gpc3.value

def peak_flux(L, dl, corr):
    # 4 * pi = 12.566
    return (L*corr) / (12.566 * dl * dl)# (1+z) *

def detect_grbs(pf, kind='sim', pf_50_300=None, func=None, z=None, L=None, \
    threshold=None):

    # GBM detection probability
    if kind == 'gbm':
        det_prob = func(pf_50_300)
        binary = rng.binomial(1, det_prob)
        detected = (binary == 1)

    # Fake GBM detection probability
    elif kind == 'sim':
        det_prob = logistic(pf, s=0.1, w=5)
        binary = rng.binomial(1, det_prob)
        detected = (binary == 1)

    # Single flux cutoff
    elif kind == 'cutoff':
        threshold = threshold
        detected = np.where(pf[:]>threshold)

    #if plot is True:
    #    plot_detections(detected, pf, z, L)
    if z is not None and L is not None:
        return pf[detected], pf_50_300[detected], z[detected], L[detected]
    else:
        return pf[detected], pf_50_300[detected]
