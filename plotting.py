import matplotlib.pyplot as plt
import numpy as np

def get_bins(data_type):
    if data_type == 'peakflux':
        return np.logspace(-6,4,80)
    elif data_type == 'duration':
        return np.logspace(-3, 4, 80)
    elif data_type == 'fluence':
        return np.logspace(-11, -2, 80)
    elif data_type == 'luminosity':
        return np.logspace(47,54,18)
    elif data_type == 'merger luminosity':
        return np.logspace(47,54,18)
    elif data_type == 'redshift':
        return np.logspace(-2,1,10)
    else:
        print ('Data type does not exist.')

def plot_luminosity_samples(sample):
    bins = get_bins('luminosity')
    h = plt.hist(sample, bins=bins)
    plt.xscale('log')
    plt.yscale('log')
    plt.grid()
    plt.show()
    plt.close()

def plot_redshift_samples(samples):
    plt.hist(samples, bins=20)
    plt.show()
    plt.close()

def plot_detections(idx, pf, z, L):
    plt.scatter(z, L,  facecolors='none', edgecolors='C0', alpha=0.1)
    plt.scatter(z[idx], L[idx], facecolors='none', edgecolors='salmon')
    plt.ylabel('Isotropic Peak Luminosity (1-10000 keV) [erg/s]')
    plt.xlabel(r'Redshift, $z$')
    plt.yscale('log')
    plt.xlabel('Redshift')
    plt.ylabel('Luminosity')
    plt.show()
    plt.close()

    plt.hist(z, bins=20)
    plt.hist(z[idx], bins=20)
    plt.xlabel('Redshift')
    plt.show()
    plt.close()

    bins = get_bins('luminosity')
    plt.hist(L, bins=bins)
    plt.hist(L[idx], bins=bins)
    plt.xscale('log')
    plt.xlabel('Luminosity')
    plt.show()
    plt.close()


def plot_data(data, type='peakflux'):
    plt.stairs(data, edges=get_bins(type))
    plt.xscale('log')
    plt.show()
    plt.close()
    return

def plot_peak_flux(pf, pf2=None):
    bins = get_bins('peakflux')
    plt.hist(pf, bins=bins, histtype='step')
    if pf2 is not None:
        plt.hist(pf2, bins=bins, histtype='step')
    plt.xscale('log')
    plt.show()
    plt.close()
    return
