
import os
from glob import glob
import sys
sys.path.insert(0, '../data/')
import matplotlib.pyplot as plt
from datacleaner import LightCurve
from results import BestLightCurve, MCMCResults, plot_star
from cleanfit import T14b2aRsi
import batman
import numpy as np

#results_dir = '/astro/store/scratch/tmp/bmmorris/stsp/kepler17/window229/run002/'

window_ind, run_ind = sys.argv[-2:]
# results_dir = ('/astro/store/scratch/tmp/bmmorris/stsp/kepler17/window{0:03d}/run{1:03d}/'
results_dir = ('/local/tmp/osg/hat11-osg/window{0:03d}/run{1:03d}/'
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

window_dir = os.sep.join(results_dir.split(os.sep)[:-2])
mcmc_paths = glob(window_dir + '/run???/*_mcmc.txt')

try:
    print(best_lc_path)
except NameError:
    raise ValueError("{0} doesn't exist.".format(results_dir))


def hat11_params():
    from hat11 import planet_properties, stellar_properties
    params = batman.TransitParams()
    params.t0 = planet_properties['first_mid_transit_time']                       #time of inferior conjunction
    params.per = planet_properties['period']                   #orbital period
    params.rp = planet_properties['transit_depth']                      #planet radius (in units of stellar radii)
    b = planet_properties["impact_parameter"]
    #inclination = 88.94560#np.arccos(b/params.a)
    params.inc = planet_properties['inclination'] #orbital inclination (in degrees)
    params.duration = planet_properties['transit_duration_days']

    ecosw = planet_properties['ecosw']
    esinw = planet_properties['esinw']
    eccentricity = np.sqrt(ecosw**2 + esinw**2)
    omega = np.degrees(np.arccos(ecosw/eccentricity))

    a, _ = T14b2aRsi(params.per, params.duration, b, params.rp, eccentricity, omega)

    params.a = a                  #semi-major axis (in units of stellar radii)

    params.ecc = eccentricity                     #eccentricity
    params.w = omega                      #longitude of periastron (in degrees)
    params.u = map(float, stellar_properties['four_param_limb_darkening'].split(' '))                #limb darkening coefficients
    params.limb_dark = "nonlinear"       #limb darkening model
    return params

transit_params = hat11_params()#get_basic_kepler17_params()
blc = BestLightCurve(best_lc_path, transit_params=transit_params)
blc.plot_whole_lc()
#blc.plot_transits()
#plt.show()

mcmc = MCMCResults(mcmc_paths)
mcmc.plot_chi2()
mcmc.plot_chains()
#mcmc.plot_star()
#mcmc.plot_corner()
#mcmc.plot_each_spot()
plt.show()
