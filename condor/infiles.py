
import os
import numpy as np
from scipy.ndimage import gaussian_filter


output_file_path = 'chunk16.in'
template_file_path = 'fill_in.in'
template_file = open(template_file_path, 'r').read()

stellar_rotation_period = np.loadtxt('../params/stellar_rotation_period.txt').item()
light_curve_path = '../lightcurves/c16.txt'
light_curve = np.loadtxt(light_curve_path)

t0 = 2454605.89135
per = 4.88780296
ecosw = 0.261
esinw = 0.085
eccentricity = np.sqrt(ecosw**2 + esinw**2)
w = np.arccos(ecosw/eccentricity)
#w += np.pi
offset = np.pi/2
ecosw = eccentricity*np.cos(w + offset)
esinw = eccentricity*np.sin(w + offset)

planet_properties = dict(n_planets=1,
                         first_mid_transit_time=t0,  # From fits in data/fit_transit.ipynb
                         period=4.88780296,  # From fits in data/fit_transit.ipynb
                         transit_depth=0.00347,  # From fits in data/fit_transit.ipynb
                         transit_duration_days=0.1031,  # From fits in data/fit_transit.ipynb
                         impact_parameter=0.278,   # From fits in data/fit_transit.ipynb
                         inclination=88.9027452976,   # From fits in data/fit_transit.ipynb
                         orbit_lambda=103,  # Winn 2010
                         ecosw=ecosw, #=0.261,  # Winn 2010
                         esinw=esinw#=0.085  # Winn 2010
                         )

stellar_properties = dict(mean_stellar_density=2.41,  # From fits in data/fit_transit.ipynb
                          stellar_rotation_period=stellar_rotation_period,  # from data/periodogram.ipynb
                          stellar_temperature=4780,  # Bakos 2010
                          stellar_metallicity=0.31,  # Bakos 2010
                          tilt_stellar_rotation_axis=80,  # Sanchis-Ojeda 2011, i_s
                          four_param_limb_darkening=' '.join(map(str, [0.000, 0.727, 0.000, -0.029])),   # From fits in data/fit_transit.ipynb
                          n_ld_rings=100
                          )

spot_properties = dict(lightcurve_path=os.path.abspath(light_curve_path),
                       start_fit_time=light_curve[0, 0],
                       fit_duration_days=light_curve[-1, 0] - light_curve[0, 0],
                       noise_corrected_max=np.max(gaussian_filter(light_curve[:, 1], 50)),
                       flattened_flag=0,
                       n_spots=4,
                       fractional_spot_contrast=0.7
                       )

# For an unseeded run:
action_properties = dict(action='M',
                         random_seed=74384338,
                         a_scale=1.25,
                         n_chains=2,
                         n_steps=100,
                         calc_brightness=1
                         )

# M		; M= unseeded mcmc
# 74384338	; random seed
# 1.25000		; ascale
# 40		; number of chains
# 5000		; mcmc steps
# 1		; 0= use downfrommax normalization, 1= calculate brightness factor for every model

assert action_properties['n_chains'] % 2 == 0, "`n_chains` must be even."
all_dicts = planet_properties
for d in [stellar_properties, spot_properties, action_properties]:
    all_dicts.update(d)
in_string = open('fill_in.in').read()

with open(output_file_path, 'w') as out:
    out.write(in_string.format(**all_dicts))
