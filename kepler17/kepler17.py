from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import os
from utils import (STSPRun, get_transit_parameters)

run_name = 'kepler17'

top_level_output_dir = os.path.join('/astro/store/scratch/tmp/bmmorris/stsp',
                                    run_name)
light_curve_path = 'kepler17_whole.dat'
parameter_file_path = 'kep17.params'
p_orb, t_0, tdur, p_rot, Rp_Rs, impact, incl_orb, Teff, sden = get_transit_parameters(parameter_file_path)

planet_properties = dict(n_planets=1,
                         first_mid_transit_time=t_0,
                         period=p_orb,
                         transit_depth=Rp_Rs,
                         transit_duration_days=tdur,
                         impact_parameter=impact,
                         inclination=incl_orb,
                         orbit_lambda=0,
                         ecosw=0.0,
                         esinw=0.0
                         )

stellar_properties = dict(mean_stellar_density=sden,
                          stellar_rotation_period=p_rot,
                          stellar_temperature=Teff,
                          stellar_metallicity=0,
                          tilt_stellar_rotation_axis=0,
                          four_param_limb_darkening=' '.join(map(str, [0.59984,
                                                                       -0.165775,
                                                                       0.6876732,
                                                                       -0.349944])),
                          n_ld_rings=100
                          )

spot_properties = dict(lightcurve_path=os.path.abspath(light_curve_path),
                       flattened_flag=0,
                       n_spots=5,
                       fractional_spot_contrast=0.7
                       )

# For an unseeded run:
action_properties = dict(random_seed=74384338,
                         a_scale=2.5,
                         n_chains=50,
                         n_steps=1000,
                         calc_brightness=1
                         )

run = STSPRun(parameter_file_path=parameter_file_path,
              light_curve_path=light_curve_path,
              executable_path='/astro/users/bmmorris/git/hat-11/kepler17/{0}.csh'.format(run_name),
              output_dir_path=top_level_output_dir,
              initial_dir=top_level_output_dir,
              condor_config_path=os.path.join('./', run_name+'.condor'),
              planet_properties=planet_properties,
              stellar_properties=stellar_properties,
              spot_properties=spot_properties,
              action_properties=action_properties,
              n_restarts=20)

#run.write_data_files()
run.create_runs()
run.make_condor_config()
