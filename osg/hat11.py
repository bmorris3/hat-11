from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import os
from utils import (STSPRun, get_transit_parameters)
from glob import glob

run_name = 'hat11'
executable_path = '/home/bmorris/git/STSP/stsp_20160125'
top_level_output_dir = os.path.join('/local-scratch/bmorris/hat11/',
                                    run_name)
transit_paths = glob('/local-scratch/bmorris/hat11/hat11_single_transits/*.txt')

planet_properties = dict(n_planets=1,
                         first_mid_transit_time=2454605.89154,
                         period=4.88780236,
                         transit_depth=0.00343**0.5,
                         transit_duration_days=0.0982,
                         impact_parameter=0.121,
                         inclination=89.45042,
                         orbit_lambda=106,
                         ecosw=0.261,
                         esinw=0.085
                         )

stellar_properties = dict(mean_stellar_density=1.81004,
                          stellar_rotation_period=29.984,
                          stellar_temperature=4780,
                          stellar_metallicity=0,
                          tilt_stellar_rotation_axis=80,
                          four_param_limb_darkening=' '.join(map(str, [0, 0.86730, 0, -0.15162])),
                          n_ld_rings=40
                          )

spot_properties = dict(lightcurve_path=None,
                       flattened_flag=1,
                       n_spots=4,
                       fractional_spot_contrast=0.7
                       )

# For an unseeded run:
action_properties = dict(random_seed=74384338,
                         a_scale=2.5,
                         n_chains=20,
                         n_steps=1000,
                         calc_brightness=1
                         )

run = STSPRun(parameter_file_path=None,
              light_curve_path=None,
              output_dir_path=top_level_output_dir,
              initial_dir=top_level_output_dir,
              condor_config_path=os.path.join('./', run_name+'.condor'),
              planet_properties=planet_properties,
              stellar_properties=stellar_properties,
              spot_properties=spot_properties,
              action_properties=action_properties,
              n_restarts=2)

run.copy_data_files(transit_paths=transit_paths)
run.create_runs()
#run.make_condor_config()
