"""
Tool for taking the raw data from MAST and producing cleaned light curves
"""

from astropy.io import fits
from glob import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import shutil
import batman

class LightCurve(object):
    """
    Container object for light curves
    """
    def __init__(self, times, fluxes, errors, quarters=None, name=None):
        self.times = times
        self.fluxes = fluxes
        self.errors = errors
        self.quarters = quarters
        self.name = name

    def plot(self, quarter=None, errors=False):
        """
        Plot light curve
        """
        if quarter is not None:
            if hasattr(quarter, '__len__'):
                mask = np.zeros_like(self.times).astype(bool)
                for q in quarter:
                    mask |= self.quarters == q
            else:
                mask = self.quarters == quarter
        else:
            mask = np.ones_like(self.times).astype(bool)

        fig, ax = plt.subplots(figsize=(8,8))
        if errors:
            ax.errorbar(self.times[mask], self.fluxes[mask], self.errors[mask])
        else:
            ax.scatter(self.times[mask], self.fluxes[mask])
        ax.set(xlabel='Time', ylabel='Flux', title=self.name)
        plt.show()

    def save_to(self, path):
        """
        Save times, fluxes, errors to new directory ``dirname`` in ``path``
        """
        dirname = self.name
        output_path = os.path.join(path, dirname)
        if os.path.exists(output_path):
            shutil.rmtree(output_path)

        os.mkdir(output_path)
        for attr in ['times', 'fluxes', 'errors', 'quarters']:
            np.save(os.path.join(path, dirname, '{}.npy'.format(attr)),
                    getattr(self, attr))

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

            polynomial = np.polyfit(self.times[quarter_mask],
                                    self.fluxes[quarter_mask], polynomial_order)
            self.fluxes[quarter_mask] /= np.polyval(polynomial,
                                                    self.times[quarter_mask])
            if plots:
                plt.plot(self.times[quarter_mask], self.fluxes[quarter_mask])
                plt.show()

        if rename is not None:
            self.name = rename

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
            times.append(data['TIME'])
            errors.append(data['PDCSAP_FLUX_ERR'])
            fluxes.append(data['PDCSAP_FLUX'])
            quarter.append(len(data['TIME'])*[header['QUARTER']])

        times, fluxes, errors, quarter = [np.concatenate(i)
                                          for i in [times, fluxes, errors, quarter]]

        mask_nans = np.zeros_like(times).astype(bool)
        for attr in [times, fluxes, errors]:
            mask_nans |= np.isnan(attr)

        times, fluxes, errors, quarter = [attr[-mask_nans]
                                           for attr in [times, fluxes, errors, quarter]]

        return LightCurve(times, fluxes, errors, quarters=quarter, name=name)

def get_basic_model():
    params = batman.TransitParams()
    params.t0 = 0.                       #time of inferior conjunction
    params.per = 1.                      #orbital period
    params.rp = 0.1                      #planet radius (in units of stellar radii)
    params.a = 15.                       #semi-major axis (in units of stellar radii)
    params.inc = 87.                     #orbital inclination (in degrees)
    params.ecc = 0.                      #eccentricity
    params.w = 90.                       #longitude of periastron (in degrees)
    params.u = [0.1, 0.3]                #limb darkening coefficients
    params.limb_dark = "quadratic"       #limb darkening model
