from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import os
from utils import (STSPRun, get_transit_parameters, quadratic_to_nonlinear_ld,
                   load_friedrich_params)
from glob import glob

import sys
import numpy as np

h11 = load_friedrich_params()

h11['t0'] += 0.25 * h11['per']

ecosw = h11['ecc'] * np.cos(np.radians(h11['w']))
esinw = h11['ecc'] * np.sin(np.radians(h11['w']))
ld_params = ' '.join(map(str, quadratic_to_nonlinear_ld(*h11['u'])))

planet_properties = dict(n_planets=1,
                         first_mid_transit_time=h11['t0'],
                         period=h11['per'],
                         transit_depth=h11['rp']**2,
                         transit_duration_days=h11['duration'],
                         impact_parameter=h11['b'],
                         inclination=h11['inc'],
                         orbit_lambda=h11['lam'],
                         ecosw=ecosw,
                         esinw=esinw
                         )

stellar_properties = dict(mean_stellar_density=h11['rho_star'],
                          stellar_rotation_period=h11['per_rot'],
                          stellar_temperature=4780,
                          stellar_metallicity=0,
                          tilt_stellar_rotation_axis=h11['inc_stellar'],
                          four_param_limb_darkening=ld_params,
                          n_ld_rings=40
                          )

spot_properties = dict(lightcurve_path=None,
                       flattened_flag=1,
                       n_spots=2,
                       fractional_spot_contrast=0.7,
                       sigma_radius=0.002,
                       sigma_angle=0.01,
                       )

n_hours = 4.0
n_seconds = int(n_hours*60*60)

# For an unseeded run:
action_properties = dict(random_seed=74384338,
                         a_scale=2.0,
                         n_chains=300,
                         n_steps=-n_seconds,
                         calc_brightness=1
                         )


if __name__ == '__main__':

    run_name = 'sun_active-osg'
    executable_path = '/home/bmorris/git/STSP/stsp_20160816'
    top_level_output_dir = os.path.join('/local-scratch/bmorris/sun_active/',
                                        run_name)
    transit_paths = glob('/local-scratch/bmorris/sun_active/friedrich/sun_active/lc*.txt')
    spot_param_paths = glob('/local-scratch/bmorris/sun_active/friedrich/sun_active/stsp_spots*.txt')

    run = STSPRun(parameter_file_path=None,
                  light_curve_path=None,
                  output_dir_path=top_level_output_dir,
                  initial_dir=top_level_output_dir,
                  condor_config_path=os.path.join('./', run_name+'.condor'),
                  planet_properties=planet_properties,
                  stellar_properties=stellar_properties,
                  spot_properties=spot_properties,
                  action_properties=action_properties,
                  n_restarts=25)

    run.copy_data_files(transit_paths=transit_paths, spot_param_paths=spot_param_paths)
    run.create_runs()
    #run.make_condor_config()
