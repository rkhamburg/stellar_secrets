"""
Script to fit simulated peak flux data with peak flux models built from
luminosity and redshift distributions.
"""
import argparse
from astropy.io import ascii
import bilby
from configparser import ConfigParser
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad
from scipy.interpolate import interp1d
from scipy.special import gammaln

from astro_calc import *
from peak_flux import GRB_Simulation
from gbm_data import *
from plotting import plot_data
from priors import make_prior_joint

class PoissonLikelihood(bilby.core.likelihood.Analytical1DLikelihood):
    def __init__(self, x, y, func):
        """
        A simple Poisson likelihood

        Parameters
        ----------
        x: array_like
            The independent data
        y: array_like
            The dependent data to analyse -- this must be a set of
            non-negative integers, each being the number of events
            within some interval
        func: function
            The python function providing the rate of events per interval to
            fit to the data. The function must be defined with the first
            argument being a dependent parameter (although this does not have
            to be used by the function if not required). The subsequent
            arguments will require priors and will be sampled over (unless a
            fixed value is given)
        """
        super(PoissonLikelihood, self).__init__(x=x, y=y, func=func)

    def log_likelihood(self):
        rate = self.func(self.x, **self.parameters)
        # Check format
        if not isinstance(rate, np.ndarray):
            raise ValueError(
                "Poisson rate function returns wrong value type! "
                "Is {} when it should be numpy.ndarray".format(type(rate)))
        # Check for negative values
        elif np.any(rate[0] < 0) or np.any(rate[1] < 0):
            raise ValueError(("Poisson rate function returns a negative",
                              " value!"))
        # Check for nan values
        elif np.any(np.isnan(rate[0])==True) or np.any(np.isnan(rate[1])==True):
            return -np.inf
        # Return likelihood
        else:
            return np.sum(-rate[0] + self.y[0]*np.log(rate[0]) \
                    - gammaln(self.y[0] + 1)) \
                 + np.sum(-rate[1] + self.y[1]*np.log(rate[1]) \
                    - gammaln(self.y[1] + 1))

    def __repr__(self):
        return bilby.core.likelihood.Analytical1DLikelihood.__repr__(self)

    @property
    def y(self):
        """ Property assures that y-value is a positive integer. """
        return self.__y

    @y.setter
    def y(self, y):
        if not isinstance(y, np.ndarray):
            y = np.array([y], dtype='object')
        # check array is a non-negative array
        if np.any(y[0] < 0) or np.any(y[1] < 0):
            raise ValueError("Data must be non-negative")
        self.__y = y


parser = argparse.ArgumentParser()
parser.add_argument('-c', default='CBC.ini', \
                    help="path to ini file")
parser.add_argument('--sim', type=bool, default=False, \
                    help='injection simulations')
parser.add_argument('-l', default='SPL', \
                    help="luminosity function: 'SPL', 'CPL', or 'BPL'")
parser.add_argument('-l2', default='SPL', \
                    help="luminosity function: 'SPL', 'CPL', or 'BPL'")
parser.add_argument('--run', type=bool, default=False, \
                    help='option to run nested sampler')
parser.add_argument('-z', default='SFR', \
                    help="redshift distribution: 'SFR' or 'WP'")
parser.add_argument('--npool', type=int, default=1,\
                    help='Number of threads')
parser.add_argument('--nlive', type=int, default=1000,\
                    help='Number of live points')
parser.add_argument('--plot', type=bool, default=False,
                    help='boolean to plot data')
args = parser.parse_args()

# Get configuration
config = ConfigParser()
config.read(args.c)
paths = config['paths']
names = config['names']
options = config['options']

# Specify the output directory and name of the simulation
outdir = names.get('outdir')
label = names.get('label')
bilby.core.utils.setup_logger(outdir=outdir, label=label)

# Set simulation number
NGRB = options.getfloat('NGRB')

# Set redshift domain
minz = options.getfloat('minz')
maxz = options.getfloat('maxz')
redshifts = np.logspace(minz, maxz, 10000)

# Set luminosity domain
minl = options.getfloat('minl')
maxl = options.getfloat('maxl')
minl_merg = options.getfloat('minl_merg')
maxl_merg = options.getfloat('maxl_merg')
coll_lum = np.logspace(minl, maxl, 10000)
merg_lum = np.logspace(minl_merg, maxl_merg, 10000)
luminosities = np.array([coll_lum, merg_lum])

# Get distance luminosities and co-moving volument elements
dl_cm = distance_luminosity(redshifts)
dV_dz_Gpc3 = diff_comoving_volume(redshifts)

# Load GBM detection threshold files
''' CHECK WHICH TIMESCALE BEING USED '''
gbm_det_prob = paths.get('gbm_1s_all')
f = ascii.read(gbm_det_prob, header_start=0)
func = interp1d(f["Flux"], f["Probability"], bounds_error=False, \
        fill_value=(0,1))

# GBM Detector Correction files
gbm_corr_10_1000      = paths.get('gbm_corr')
gbm_corr_50_300       = paths.get('gbm_corr_50_300')
gbm_corr_merg_10_1000 = paths.get('gbm_corr_merg')
gbm_corr_merg_50_300  = paths.get('gbm_corr_merg_50_300')
collapsar_correction        = np.load(gbm_corr_10_1000)
collapsar_correction_50_300 = np.load(gbm_corr_50_300)
merger_correction           = np.load(gbm_corr_merg_10_1000)
merger_correction_50_300    = np.load(gbm_corr_merg_50_300)

# Get injection parameters
keys = [p for p in options['parameter_labels'].split(',')]
values = [float(p) for p in options['parameter_injections'].split(',')]
injection_parameters = dict(zip(keys, values))

#from source_distributions import Redshift
#M = Redshift(redshifts, None)
#merger_rate = M.merger_rate_density(cosmo, save=True)
merger_rate = paths.get('merger_rate')
merger_rate = np.load(merger_rate)

# Generate injected data or read in real GBM data
if args.sim is not False:
    data = GRB_Simulation([redshifts,
                            luminosities,
                            collapsar_correction,
                            collapsar_correction_50_300,
                            merger_correction,
                            merger_correction_50_300,
                            func,
                            dl_cm,
                            dV_dz_Gpc3,
                            merger_rate,
                            NGRB,
                            args.l, args.z, args.l2],
                            **injection_parameters
    )

else:
    gbm_pf_file     = paths.get('t90_file')
    luminosity_file = paths.get('rest_frame_file')
    pf_data = get_data(gbm_pf_file, type='all')[1]
    lum_data = get_luminosity_data(luminosity_file, t90_file=gbm_pf_file,
        type='all')
    data = np.array([pf_data, lum_data], dtype=object)

# Plot data
''' NEEDS FIXING '''
if args.plot is not False:
    plot_data(data, type='peakflux')

likelihood = PoissonLikelihood([redshifts,
                                luminosities,
                                collapsar_correction,
                                collapsar_correction_50_300,
                                merger_correction,
                                merger_correction_50_300,
                                func,
                                dl_cm,
                                dV_dz_Gpc3,
                                merger_rate,
                                NGRB,
                                args.l, args.z, args.l2],
                                data, GRB_Simulation
)


# Define Priors
priors = make_prior_joint(args.l, args.z, args.l2)

# Run nested sampler
if args.run is not False:

    if args.sim is not False:
        result = bilby.run_sampler(likelihood=likelihood, priors=priors,
        sampler='dynesty', sample='unif', npool=args.npool, npoints=args.nlive,
        outdir=outdir, label=label, injection_parameters=injection_parameters)
    else:
        result = bilby.run_sampler(likelihood=likelihood, priors=priors,
        sampler='dynesty', sample='unif', npool=args.npool, npoints=args.nlive,
        outdir=outdir, label=label)
