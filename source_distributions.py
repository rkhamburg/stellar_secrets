#from math_functions import log_normal, lognorm_params
import numpy as np
from scipy.integrate import quad, simpson
from astropy.cosmology import z_at_value
import astropy.units as u
import matplotlib.pyplot as plt

"""
class Duration():
    def __init__(self, durations, mode, stddev):
        self.d = durations
        self.m = mode
        self.s = stddev
        return

    def intrinsic_duration(self):
        '''
        T90 is usually calculated from fluence between 50-300 keV.
        We might have to modify our comparisons because I'm taking
        flux between 10-1000 keV (maybe higher).
        '''
        # Convert mode and std to mu and sigma of underlying normal
        # distribution to input into scipy
        sigma, scale = lognorm_params(self.m, self.s)
        mu = np.log(scale)
        # Un-normalized duration pdf (i.e., t90_freq does not sum to 1)
        t90_freq = log_normal(self.d, mu=mu, sigma=sigma, A=1.)
        return t90_freq

    def cosmo_expand(self, dur, z):
        return dur * (1+z)
"""

class Luminosity():
    def __init__(self, l, L0, L1=None, L2=None, L3=None, L4=None, lmin=1E49, \
    lmax=1E55, name='SPL'):
        """
        Parameters:
        -----------
        l: array
            Luminosity domain (log10)
        L0: float
            lower index
        L1: float
            upper index or luminosity cutoff
        L2: float
            luminosity break
        L3: float
            lowest index (for DBPL)
        L4: float
            low luminosity break (for DBPL)
        lmin: float
            minimum luminosity
        lmax: float
            maximum luminosity
        """
        self.l = l
        self.L0 = L0
        self.L1 = L1
        self.L2 = L2
        self.L3 = L3
        self.min = lmin
        self.max = lmax
        self.f = name
        return

    def DBPL(self):
        # Break luminosities into 3 domains
        lum0 = self.l[self.l <= 1E49]
        lum1 = self.l[self.l <= self.L2]
        lum1 = lum1[lum1 >= 1E49]
        lum2 = self.l[self.l > self.L2]

        # Luminosity function
        phi = lambda l, index, lbreak: (l/lbreak) ** index
        phi2 = lambda l, index1, index2, lbreak1, lbreak2: \
            ((l/lbreak1) ** index1) * ((lbreak1/lbreak2) ** index2)

        # Evaluate luminosity function
        phi0 = phi2(lum0, self.L3, self.L0, 1E49, self.L2)
        phi1 = phi(lum1, self.L0, self.L2)
        phi2 = phi(lum2, self.L1, self.L2)
        return np.concatenate((phi0,phi1,phi2))


    def BPL(self, plot=False):
        """ Broken power law from W&P 2010
        """
        # Break luminosities into 2 domains
        lum1 = self.l[self.l <= self.L2]
        lum2 = self.l[self.l > self.L2]

        # Luminosity function
        phi = lambda l, index, lbreak: (l/lbreak) ** index

        # Evaluate luminosity function
        phi1 = phi(lum1, self.L0, self.L2)
        phi2 = phi(lum2, self.L1, self.L2)

        # Integrate luminosity function
        #A = quad(phi, self.min, self.L2, args=(self.L0, self.L2))[0] \
        #  + quad(phi, self.L2, self.max, args=(self.L1, self.L2))[0]

        # Normalize luminosity function
        phi1 = phi1 #/ A
        phi2 = phi2 #/ A
        return np.concatenate((phi1,phi2))


    def CPL(self):
        # Luminosity function
        phi = lambda l, index, Lc: ((l/Lc) ** index) * np.exp(-l/Lc)

        # Evaluate luminosity function
        phi_eval = phi(self.l, self.L0, self.L2)

        # Integrate luminosity function
        #A = quad(phi, self.min, self.max, args=(self.L0, self.L2,))[0]
        return phi_eval #/ A


    def SPL(self):
        """ Single power law function normalized by a break luminosity
        """
        # Luminosity Function
        phi = lambda l, index: l ** index

        # Evaluate luminosity function
        phi_eval = phi(self.l, self.L0)

        # Integrate luminosity function
        #A = quad(phi, self.min, self.max, args=(self.L0,))[0]
        return phi_eval #/ A


    def set_luminosity_function(self):
        if self.f == 'SPL':
            return self.SPL()
        elif self.f == 'CPL':
            return self.CPL()
        elif self.f == 'BPL':
            return self.BPL()
        elif self.f == 'DBPL':
            return self.DBPL()

    def luminosity_pdf(self):
        luminosity_function = self.set_luminosity_function()
        #self.plot_luminosity_dist(luminosity_function, function=True)
        luminosity_density = luminosity_function * self.l
        #self.plot_luminosity_dist(luminosity_density)
        return luminosity_density


    def plot_luminosity_dist(self, pdf, function=False):
        plt.plot(self.l, pdf)
        plt.xscale('log')
        if function is not False:
            plt.yscale('log')
            plt.ylabel('dN dL')
            plt.grid()
        else:
            plt.ylabel('dN')
        plt.xlabel('Isotropic Peak Luminosity [erg/s]')
        plt.show()
        plt.close()


class Redshift():
    def __init__(self, z, r0, r1=None, r2=None, r3=None, name=None):
        """
        Parameters
        ----------
        f: string
            title of distribution to be used
        """
        self.r0 = r0
        self.r1 = r1
        self.r2 = r2
        self.r3 = r3
        self.z = z
        self.f = name
        return

    def set_redshift_function(self):
        if self.f == 'SFR':
            return self.MadauFragos2017()
        elif self.f == 'WP':
            return self.WP2010()

    def HopkinsBeacom2006(self):
        return (0.7 * (0.017 + 0.13*self.z)) / (1 + (self.z/3.3)**5.3)

    def Li2008(self):
        """
        SFR using UV and IR in units of [M_* yr^-1 Mpc^-3]
        But we normalize it to [Gpc^-3 yr^-1] using the
        local density

        Here:
            r0 = local rate density [Gpc^-3 yr^-1]

        Returns:
        -----------
        Rate of GRBs at the given redshift
        """
        return (0.0157 + 0.12*self.z) / (1 + (self.z/3.23)**4.66)

    def MadauDickinson2014(self, z=None):
        """
        SFR using UV and IR in units of [M_* yr^-1 Mpc^-3]
        But we normalize it to [Gpc^-3 yr^-1] using the
        local density

        Here:
            r0 = local rate density [Gpc^-3 yr^-1]

        Returns:
        -----------
        Rate of GRBs at the given redshift
        """
        return 0.015 * ( (1 + self.z)**2.7 / (1 + ((1+self.z)/2.9)**5.6) )

    def MadauFragos2017(self, z=None):
        """
        SFR using UV and IR in units of [M_* yr^-1 Mpc^-3]
        But we normalize it to [Gpc^-3 yr^-1] using the
        local density

        Here:
            r0 = local rate density [Gpc^-3 yr^-1]

        Returns:
        -----------
        Rate of GRBs at the given redshift
        """
        return 0.01 * (1+self.z)**2.6 / (1 + ((1+self.z)/3.2)**6.2)

    def WP2010(self):
        """ Long GRB Rate distribution from Wandermann and Piran 2010

        Here:
            r0 = local rate density
            r1 = low redshift index
            r2 = high redshift index
            r3 = redshift break
        Returns:
        ------------
        r: float
            Rate of GRBs at the given redshift
        """
        return np.piecewise(self.z, [self.z < self.r3, self.z >= self.r3], \
         [lambda z: self.r0 * (1+z)**self.r1, \
          lambda z: self.r0 * ((1+self.r3)**(self.r1-self.r2)) * (1+z)**self.r2])


    def source_rate_density(self, dV, merger_rate=None):
        """ Calculates observed GRB redshift distribution

        Parameters:
        ------------
        redshifts: array
            list or np array of redshifts

        Returns:
        rate_pdf:
        """
        # GRB Rate Density [Gpc^-3 yr^-1]
        if merger_rate is None:
            rgrb = self.set_redshift_function()
            rgrb = self.r0 * rgrb / rgrb[0]
            #rgrb = self.WP2010()
        else:
            rgrb = self.r0 * merger_rate / merger_rate[0]
        #self.plot_intrinsic_rate(rgrb)

        # Redshift Distribution
        # [GRBs / year / dz / dOmega]
        num = len(rgrb)
        rate = rgrb * dV[:num] / (1+self.z[:num])

        # Total GRBs in universe / year / dOmega
        N = simpson(rate, self.z[:num])

        # Normalize rate for sampling
        rate_pdf = rate[:-1] * np.diff(self.z)
        #self.plot_observed_rate(rate_pdf)
        rate_pdf /= np.sum(rate_pdf)
        return rate_pdf, N


    def merger_rate_density(self, cosmo, tmin=20, save=False):
        """ Calculates intrinsic merger redshift distribution.
            The merger rate at redshift z is equal to SFR
            contributions (i.e., when mergers are formed)
            multiplied by the probability distribution of
            time delays, which I assume to be ~ 1 / delta T.
            A minimum time delay is assumed.

        Parameters:
        ------------
        look_back_times: array or list
            array of the lookback times calculated for each redshift
        tmin: int or float
            minumum time delay between binary formation and merging

        Returns:
        ------------
        merger_rate: array
            rate of mergers at every redshift in the domain

        """
        # MadauFragos2017 [M_* Gpc^-3 yr^-1]
        sfr = lambda z: 1E7 * (1+z)**2.6 / (1 + ((1+z)/3.2)**6.2)

        # Cosmo
        tmax = 13400 #cosmo.hubble_time.to(u.Myr).value
        look_back_times = cosmo.lookback_time(self.z).to(u.Myr).value
        time_delays = np.linspace(tmin, tmax, 1000)

        # BNS rate density
        rate = []
        for i in range(len(self.z)):

            # Lookback time and redshift at which binaries merged
            time_merg = look_back_times[i]
            print (self.z[i], time_merg)

            # Calculate contributions from forming binaries
            r = []
            for j in range(len(time_delays)):

                # Time of binary formation
                tf = np.min([time_delays[j] + time_merg, tmax])
                if tf == tmax:
                    break
                # Redshift of binary formation
                zf = z_at_value(cosmo.lookback_time, tf * u.Myr, zmin=0.003)
                # SFR * P(time delay)
                r.append(sfr(zf) * time_delays[j]**-1)

            # Integrate over all time delays
            rate.append(simpson(r, time_delays[:len(r)]))

        #plot_intrinsic_rate(rate)
        if save is not False:
            np.save('bns_rate_pl_20.npy', rate)
        return np.array(rate)



    def plot_intrinsic_rate(self, rgrb):
        """
        # Intrinisc rate of GRBs [Gpc^-3 yr^-1]
        """
        plt.plot(self.z[:len(rgrb)], rgrb)
        plt.yscale('log')
        plt.xlabel('Redshift')
        plt.ylabel(r'GRB Rate [Gpc$^{-3}$ yr$^{-1}$]')
        plt.show()

    def plot_observed_rate(self, rate):
        """
        Observed rate of GRBs / year / dz
        """
        plt.plot(self.z[:len(rate)], rate)
        plt.xlabel('Redshift')
        plt.ylabel(r'Observed GRB Rate')
        plt.show()
