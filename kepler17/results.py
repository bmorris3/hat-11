
import os
from glob import glob
import sys
sys.path.insert(0, '../data/')
import matplotlib.pyplot as plt
from datacleaner import LightCurve
from results import BestLightCurve, MCMCResults, plot_star
import batman
import numpy as np

#results_dir = '/astro/store/scratch/tmp/bmmorris/stsp/kepler17/window229/run002/'

window_ind, run_ind = sys.argv[-2:]
results_dir = ('/astro/store/scratch/tmp/bmmorris/stsp/kepler17/window{0:03d}/run{1:03d}/'
               .format(int(window_ind), int(run_ind)))
#if not os.path.exists(results_dir):

print('Results from: {0}'.format(results_dir))
#results_dir = '/astro/store/scratch/tmp/bmmorris/stsp/kepler17/window101/run009/'
#results_dir = os.path.abspath('../condor/tmp/')

files_in_dir = glob(os.path.join(results_dir, '*.txt'))

for output_file in files_in_dir:
    if output_file.endswith('_errstsp.txt'):
        error_path = output_file
    elif output_file.endswith('_finalparam.txt'):
        final_params_path = output_file
    elif output_file.endswith('_lcbest.txt'):
        best_lc_path = output_file
    elif output_file.endswith('_mcmc.txt'):
        mcmc_path = output_file
    elif output_file.endswith('_parambest.txt'):

        best_params_path = output_file
try:
    print(best_lc_path)
except NameError:
    raise ValueError("{0} doesn't exist.".format(results_dir))

def get_basic_kepler17_params():
    # http://exoplanets.org/detail/HAT-P-11_b
    params = batman.TransitParams()
    params.t0 = 2455185.678035                       #time of inferior conjunction
    params.per = 1.4857108                     #orbital period
    params.rp = 0.13413993                      #planet radius (in units of stellar radii)
    b = 0.1045441
    #inclination = 88.94560#np.arccos(b/params.a)
    params.inc = 88.94560 #orbital inclination (in degrees)
    inclination = np.radians(params.inc)
    params.inclination = params.inc
    params.a = b/np.cos(inclination)                       #semi-major axis (in units of stellar radii)
    params.ecc = 0.                      #eccentricity
    params.w = 90.                       #longitude of periastron (in degrees)
    params.u = [0.1, 0.3]                #limb darkening coefficients
    params.limb_dark = "quadratic"       #limb darkening model
    params.duration = params.per/np.pi*np.arcsin(np.sqrt((1+params.rp)**2 + b**2)
                                                 / np.sin(inclination)/params.a)
    return params

transit_params = get_basic_kepler17_params()
blc = BestLightCurve(best_lc_path, transit_params=transit_params)
blc.plot_whole_lc()
blc.plot_transits()
#plt.show()

mcmc = MCMCResults(mcmc_path)
mcmc.plot_chains()
mcmc.plot_star()
plt.show()
