"""
Tool for taking the raw data from MAST and producing cleaned light curves
"""

from astropy.io import fits
from astropy.time import Time
import astropy.units as u
import os
import numpy as np
import matplotlib.pyplot as plt
import shutil
import batman
from scipy import optimize

def get_basic_HAT11_params():
    # http://exoplanets.org/detail/HAT-P-11_b
    params = batman.TransitParams()
    params.t0 = 2454605.89132                       #time of inferior conjunction
    params.per = 4.8878018                     #orbital period
    params.rp = 0.00332**0.5                      #planet radius (in units of stellar radii)
    params.a = 15.01                       #semi-major axis (in units of stellar radii)
    b = 0.35
    inclination = np.arccos(b/params.a)
    params.inc = np.degrees(inclination) #orbital inclination (in degrees)
    params.ecc = 0.                      #eccentricity
    params.w = 90.                       #longitude of periastron (in degrees)
    params.u = [0.1, 0.3]                #limb darkening coefficients
    params.limb_dark = "quadratic"       #limb darkening model
    params.duration = params.per/np.pi*np.arcsin(np.sqrt((1+params.rp)**2 + b**2)
                                                 / np.sin(inclination)/params.a)
    params.inclination = inclination
    return params

def T14b2aRsi(P, T14, b):
    '''
    Convert from duration and impact param to a/Rs and inclination
    '''
    i = np.arccos( ( (P/np.pi)*np.sqrt(1 - b**2)/(T14*b) )**-1 )
    aRs = b/np.cos(i)
    return aRs, np.degrees(i)

def generate_fiducial_model_lc_short(times, t0, depth, dur, b):
    # LD parameters from Deming 2011 http://adsabs.harvard.edu/abs/2011ApJ...740...33D
    rp = depth**0.5
    exp_time = (1*u.min).to(u.day).value
    params = batman.TransitParams()
    params.t0 = t0                       #time of inferior conjunction
    params.per = 4.8878018                      #orbital period
    params.rp = rp                      #planet radius (in units of stellar radii)

    a, inc = T14b2aRsi(params.per, dur, b)

    params.a = a                       #semi-major axis (in units of stellar radii)
    params.inc = inc #orbital inclination (in degrees)
    params.ecc = 0                      #eccentricity
    params.w = 90.                       #longitude of periastron (in degrees)
    params.u = [0.6136, 0.1062]                #limb darkening coefficients
    params.limb_dark = "quadratic"       #limb darkening model

    m = batman.TransitModel(params, times, supersample_factor=7,
                            exp_time=exp_time)
    model_flux = m.light_curve(params)
    return model_flux


# Set transit parameter defaults to the HAT-P-11 values on exoplanets.org
params = get_basic_HAT11_params()

class LightCurve(object):
    """
    Container object for light curves
    """
    def __init__(self, times=None, fluxes=None, errors=None, quarters=None, name=None):

        if isinstance(times[0], Time) and isinstance(times, np.ndarray):
            times = Time(times)
        elif not isinstance(times, Time):
            times = Time(times, format='jd')

        self.times = times
        self.fluxes = fluxes
        self.errors = errors
        self.quarters = quarters
        self.name = name

    def plot(self, quarter=None, plot_date=False, save_path=None, show=True):
        """
        Plot light curve
        """
        if quarter is not None:
            if hasattr(quarter, '__len__'):
                mask = np.zeros_like(self.fluxes).astype(bool)
                for q in quarter:
                    mask |= self.quarters == q
            else:
                mask = self.quarters == quarter
        else:
            mask = np.ones_like(self.fluxes).astype(bool)

        fig, ax = plt.subplots(figsize=(8,8))

        if plot_date:
            ax.plot_date(self.times[mask].plot_date, self.fluxes[mask], 'o')
        else:
            ax.plot(self.times[mask].jd, self.fluxes[mask], 'o')
        ax.set(xlabel='Time', ylabel='Flux', title=self.name)

        if save_path is not None:
            fig.savefig(save_path, bbox_inches='tight')

        if show:
            plt.show()
        else:
            plt.close()

    def save_to(self, path, overwrite=False):
        """
        Save times, fluxes, errors to new directory ``dirname`` in ``path``
        """
        dirname = self.name
        output_path = os.path.join(path, dirname)
        self.times = Time(self.times)

        if os.path.exists(output_path) and overwrite:
            shutil.rmtree(output_path)

        if not os.path.exists(output_path):
            os.mkdir(output_path)
            for attr in ['times', 'fluxes', 'errors', 'quarters']:
                np.save(os.path.join(path, dirname, '{}.npy'.format(attr)),
                        getattr(self, attr))

    @classmethod
    def from_raw_fits(cls, fits_paths, name=None):
        """
        Load FITS files from MAST into the LightCurve object
        """
        fluxes = []
        errors = []
        times = []
        quarter = []

        for path in fits_paths:
            data = fits.getdata(path)
            header = fits.getheader(path)
            times.append(data['TIME'] + 2454833.0)
            errors.append(data['PDCSAP_FLUX_ERR'])
            fluxes.append(data['PDCSAP_FLUX'])
            quarter.append(len(data['TIME'])*[header['QUARTER']])

        times, fluxes, errors, quarter = [np.concatenate(i)
                                          for i in [times, fluxes, errors, quarter]]

        mask_nans = np.zeros_like(fluxes).astype(bool)
        for attr in [times, fluxes, errors]:
            mask_nans |= np.isnan(attr)

        times, fluxes, errors, quarter = [attr[-mask_nans]
                                           for attr in [times, fluxes, errors, quarter]]

        return LightCurve(times, fluxes, errors, quarters=quarter, name=name)

    @classmethod
    def from_dir(cls, path):
        """Load light curve from numpy save files in ``dir``"""
        times, fluxes, errors, quarters = [np.load(os.path.join(path, '{}.npy'.format(attr)))
                                           for attr in ['times', 'fluxes', 'errors', 'quarters']]

        if os.sep in path:
            name = path.split(os.sep)[-1]
        else:
            name = path
        return cls(times, fluxes, errors, quarters=quarters, name=name)

    def normalize_each_quarter(self, rename=None, polynomial_order=2, plots=False):
        """
        Use 2nd order polynomial fit to each quarter to normalize the data
        """
        quarter_inds = list(set(self.quarters))
        quarter_masks = [quarter == self.quarters for quarter in quarter_inds]

        for quarter_mask in quarter_masks:

            polynomial = np.polyfit(self.times[quarter_mask].jd,
                                    self.fluxes[quarter_mask], polynomial_order)
            scaling_term = np.polyval(polynomial, self.times[quarter_mask].jd)
            self.fluxes[quarter_mask] /= scaling_term
            self.errors[quarter_mask] /= scaling_term

            if plots:
                plt.plot(self.times[quarter_mask], self.fluxes[quarter_mask])
                plt.show()

        if rename is not None:
            self.name = rename

    def mask_out_of_transit(self, params=params):
        """
        Mask out the out-of-transit light curve based on transit parameters
        """
        # Fraction of one duration to capture out of transit
        get_oot_duration_fraction = 0.25
        phased = (self.times.jd - params.t0) % params.per
        near_transit = ((phased < params.duration*(0.5 + get_oot_duration_fraction)) |
                        (phased > params.per - params.duration*(0.5 + get_oot_duration_fraction)))
        sort_by_time = np.argsort(self.times[near_transit].jd)
        return dict(times=self.times[near_transit][sort_by_time],
                    fluxes=self.fluxes[near_transit][sort_by_time],
                    errors=self.errors[near_transit][sort_by_time],
                    quarters=self.quarters[near_transit][sort_by_time])

    def get_transit_light_curves(self, params=params, plots=False):
        """
        For a light curve with transits only (returned by get_only_transits),
        split up the transits into their own light curves, return a list of
        `TransitLightCurve` objects
        """
        time_diffs = np.diff(sorted(self.times.jd))
        diff_between_transits = params.per/2.
        split_inds = np.argwhere(time_diffs > diff_between_transits) + 1

        split_ind_pairs = [[0, split_inds[0][0]]]
        split_ind_pairs.extend([[split_inds[i][0], split_inds[i+1][0]]
                                 for i in range(len(split_inds)-1)])
        split_ind_pairs.extend([[split_inds[-1], len(self.times)]])

        transit_light_curves = []
        counter = -1
        for start_ind, end_ind in split_ind_pairs:
            counter += 1
            if plots:
                plt.plot(self.times.jd[start_ind:end_ind],
                         self.fluxes[start_ind:end_ind], '.-')

            parameters = dict(times=self.times[start_ind:end_ind],
                              fluxes=self.fluxes[start_ind:end_ind],
                              errors=self.errors[start_ind:end_ind],
                              quarters=self.quarters[start_ind:end_ind],
                              name=counter)
            transit_light_curves.append(TransitLightCurve(**parameters))
        if plots:
            plt.show()

        return transit_light_curves


class TransitLightCurve(LightCurve):
    """
    Container for a single transit light curve
    """
    def __init__(self, times=None, fluxes=None, errors=None, quarters=None, name=None):
        if isinstance(times[0], Time) and isinstance(times, np.ndarray):
            times = Time(times)
        elif not isinstance(times, Time):
            times = Time(times, format='jd')
        self.times = times
        self.fluxes = fluxes
        self.errors = errors
        self.quarters = quarters
        self.name = name

    def remove_linear_baseline(self, plots=False):
        """
        Find OOT portions of transit light curve using similar method to
        `LightCurve.mask_out_of_transit`, fit linear baseline to OOT,
        divide whole light curve by that fit
        """
        get_oot_duration_fraction = 0
        phased = (self.times.jd - params.t0) % params.per
        near_transit = ((phased < params.duration*(0.5 + get_oot_duration_fraction)) |
                        (phased > params.per - params.duration*(0.5 + get_oot_duration_fraction)))

        if plots:
            fig, ax = plt.subplots(1, 2, figsize=(15,6))
            ax[0].axhline(1, ls='--', color='k')
            ax[0].plot(self.times.jd, self.fluxes, 'o')
            ax[0].plot(self.times.jd[near_transit], self.fluxes[near_transit], 'ro')
            ax[0].set_title('before trend removal')

        # Remove linear baseline trend
        order = 1
        linear_baseline = np.polyfit(self.times.jd[-near_transit],
                                     self.fluxes[-near_transit], order)
        linear_baseline_fit = np.polyval(linear_baseline, self.times.jd)
        self.fluxes =  self.fluxes/linear_baseline_fit
        self.errors = self.errors/linear_baseline_fit

        if plots:
            ax[1].set_title('after trend removal')
            ax[1].axhline(1, ls='--', color='k')
            ax[1].plot(self.times.jd, self.fluxes, 'o')
            plt.show()

    def fiducial_transit_fit(self, plots=False):
        # Determine cadence:
        typical_time_diff = np.median(np.diff(self.times.jd))*u.day
        exp_long = 30*u.min
        exp_short = 1*u.min
        exp_time = (exp_long if np.abs(typical_time_diff - exp_long) < 1*u.min
                    else exp_short).to(u.day).value

        # [t0, rp, dur, b]
        initial_parameters = [2454605.89155, 0.003365, 0.092, 0.307]
        def minimize_this(p, times, fluxes, errors):
            return np.sum(((generate_fiducial_model_lc_short(times, *p) - fluxes)/errors)**2)
        fit_result = optimize.fmin(minimize_this, initial_parameters,
                                   args=(self.times.jd, self.fluxes, self.errors),
                                   disp=False)
        p = fit_result#[0]#fit_result[0]
        init_model = generate_fiducial_model_lc_short(self.times.jd, *initial_parameters)
        model_flux = generate_fiducial_model_lc_short(self.times.jd, *p)

        if plots:
            plt.plot(self.times.jd, init_model, 'g')
            plt.errorbar(self.times.jd, self.fluxes, self.errors, fmt='.')
            plt.plot(self.times.jd, model_flux, 'r')
            plt.show()

        chi2 = np.sum((self.fluxes - model_flux)**2/self.errors**2)
        return p, chi2

    @classmethod
    def from_dir(cls, path):
        """Load light curve from numpy save files in ``dir``"""
        times, fluxes, errors, quarters = [np.load(os.path.join(path, '{}.npy'.format(attr)))
                                           for attr in ['times', 'fluxes', 'errors', 'quarters']]

        if os.sep in path:
            name = path.split(os.sep)[-1]
        else:
            name = path
        return cls(times, fluxes, errors, quarters=quarters, name=name)

def combine_light_curves(light_curve_list, name=None):
    """
    Phase fold transits
    """
    times = []
    fluxes = []
    errors = []
    quarters = []
    for light_curve in light_curve_list:
        times.append(light_curve.times.jd)
        fluxes.append(light_curve.fluxes)
        errors.append(light_curve.errors)
        quarters.append(light_curve.quarters)
    times, fluxes, errors, quarters = [np.concatenate(i)
                                       for i in [times, fluxes, errors, quarters]]

    times = Time(times, format='jd')
    return TransitLightCurve(times=times, fluxes=fluxes, errors=errors,
                      quarters=quarters, name=name)

