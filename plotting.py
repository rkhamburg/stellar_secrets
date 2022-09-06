import matplotlib.pyplot as plt
import numpy as np
import corner

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


def plot_data(data, dists=None, type='peakflux'):
    plt.stairs(data, edges=get_bins(type))
    if dists is not None:
        for d in dists:
            plt.stairs(d, edges=get_bins(type))
    if type == 'peakflux':
        plt.xlim(left=0.1)
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



def corner_plots(results, bins=20, map=None, medians=None, truths=None,
    labels=None, save=False, name=None, smoothing=None, hcolor='black',
    truth_color='tomato'):
    """ Make corner plot of posterior results """

    figure = corner.corner(results, bins=bins, smooth=smoothing,
        truths=truths, truth_color=truth_color, color=hcolor,
        labels=labels, label_kwargs={'fontsize':14},
        quantiles=[0.05,0.95], show_titles=True,
        title_quantiles=[0.05,0.5,0.95], title_kwargs={"fontsize": 14},
        plot_datapoints=True)#, levels=[0.393, 0.86466, 0.98889], fill_contours=True)

    if map is not None:
        map = results[-1]
    ndim = len(map)
    axes = np.array(figure.axes).reshape((ndim, ndim))

    if truths is None:
        for i in range(ndim):
            ax = axes[i, i]
            #ax.axvline(map[i], color=truth_color, linestyle='--')
            ax.axvline(medians[i], color=truth_color)

        # Loop over the histograms
        for yi in range(ndim):
            for xi in range(yi):
                ax = axes[yi, xi]
                #ax.axvline(map[xi], color=truth_color, linestyle='--')
                #ax.axhline(map[yi], color=truth_color, linestyle='--')
                #ax.plot(map[xi], map[yi], "s", color=truth_color)
                ax.axvline(medians[xi], color=truth_color)
                ax.axhline(medians[yi], color=truth_color)
                ax.plot(medians[xi], medians[yi], "s", color=truth_color)

    axes = figure.axes
    for a in range(len(axes)):
        axes[a].tick_params(axis='x', labelsize=12)
        axes[a].tick_params(axis='y', labelsize=12)

    if save is not False:
        plt.savefig(name+'.pdf')#, dpi=600)
    else:
        plt.show()
    return
