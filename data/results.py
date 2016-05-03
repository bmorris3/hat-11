
import os
import numpy as np
from glob import glob
import matplotlib
#matplotlib.use('agg')
from matplotlib import pyplot as plt
import astropy.units as u
from astropy.constants import R_sun
from astropy.time import Time
from astropy.coordinates import SphericalRepresentation, CartesianRepresentation
from datacleaner import LightCurve
import corner

results_dir = os.path.abspath('../condor/tmp_long/')
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


class BestLightCurve(object):
    def __init__(self, path, transit_params=None):
        self.path = path
        self.default_figsize = (12, 8)#(20, 8)

        times, fluxes_kepler, errors, fluxes_model, flags = np.loadtxt(path, unpack=True)

        self.times = times if times.mean() > 2450000 else times + 2454833.
        self.fluxes_kepler = fluxes_kepler
        self.errors = errors
        self.fluxes_model = fluxes_model
        self.flags = flags

        self.kepler_lc = LightCurve(times=self.times, fluxes=fluxes_kepler, errors=errors)
        self.model_lc = LightCurve(times=self.times, fluxes=fluxes_model)
        self.transit_params = transit_params

    def plot_whole_lc(self):

        # Whole light curve
        fig, ax = plt.subplots(2, 1, figsize=self.default_figsize,
                               sharex='col')
        ax[0].plot_date(self.kepler_lc.times.plot_date, self.fluxes_kepler,
                        'k.', label='Kepler')
        ax[0].plot_date(self.model_lc.times.plot_date, self.fluxes_model,
                        'r', label='STSP')
        #ax[0].legend(loc='lower left')
        ax[0].set(ylabel='Flux')

        ax[1].plot_date(self.kepler_lc.times.plot_date,
                        self.fluxes_kepler - self.fluxes_model, 'k.')
        ax[1].set(xlabel='Time', ylabel='Residuals')
        ax[1].axhline(0, color='r')
        #
        # for l in ax[1].get_xticklabels():
        #     l.set_rotation(45)
        #     l.set_ha('right')

        return fig, ax

    def plot_transits(self):

        if self.transit_params is not None:
            kepler_transits = LightCurve(**self.kepler_lc.mask_out_of_transit(params=self.transit_params)
                                         ).get_transit_light_curves(params=self.transit_params)
            model_transits = LightCurve(**self.model_lc.mask_out_of_transit(params=self.transit_params)
                                        ).get_transit_light_curves(params=self.transit_params)

        else:
            kepler_transits = LightCurve(**self.kepler_lc.mask_out_of_transit()
                                         ).get_transit_light_curves()
            model_transits = LightCurve(**self.model_lc.mask_out_of_transit()
                                        ).get_transit_light_curves()

        # Whole light curve

        if len(kepler_transits) > 0:
            fig, ax = plt.subplots(2, len(kepler_transits), figsize=self.default_figsize,
                                   sharex='col')
            scale_factor = 0.4e-6
            for i in range(len(kepler_transits)):
                ax[0, i].plot_date(kepler_transits[i].times.plot_date, scale_factor*kepler_transits[i].fluxes,
                              'k.', label='Kepler')
                ax[0, i].plot_date(model_transits[i].times.plot_date, scale_factor*model_transits[i].fluxes,
                              'r', label='STSP')
                ax[0, i].set(yticks=[])

                ax[1, i].axhline(0, color='r', lw=2)

                ax[1, i].plot_date(kepler_transits[i].times.plot_date,
                                   scale_factor*(kepler_transits[i].fluxes -
                                            model_transits[i].fluxes), 'k.')
                xtick = Time(kepler_transits[i].times.jd.mean(), format='jd')
                #ax[1, i].set(xticks=[xtick.plot_date], xticklabels=[xtick.iso.split('.')[0]],
                #             yticks=[])

                #ax[1, i].set(xlabel='Time')

            #ax[0, 0].legend(loc='lower left')
            ax[0, 0].set(ylabel=r'Flux')# $\times \, 10^{{{0:.1f}}}$'.format(np.log10(scale_factor)))
            ax[1, 0].set(ylabel=r'Residuals')# $\times \, 10^{{{0:.1f}}}$'.format(np.log10(scale_factor)))
            #fig.subplots_adjust(wspace=0.5)
            fig.tight_layout()
            return fig, ax
        else:
            return None, None


class MCMCResults(object):
    def __init__(self, paths, burnin=0.5):

        self.table = []
        self.chain_ind = []
        self.burnin = burnin
        #self.step_ind = []

        for path in paths:
            print(path)
            table = np.loadtxt(path)
            if len(self.table) == 0:
                self.table = table
            else:
                self.table = np.vstack([self.table, table])

            self.chain_ind = np.concatenate([self.chain_ind, table[:, 0]])

            #offset = 0 if len(self.step_ind) == 0 else self.step_ind[-1]
            #plt.figure()
            #plt.plot(self.step_ind)
            #self.step_ind = np.concatenate([self.step_ind, table[:, 1] + offset])

        self.n_properties_per_spot = 3
        self.col_offset = 4
        self.n_spots = (self.table.shape[1] - self.col_offset)/self.n_properties_per_spot

        # Note: latitude is defined on (0, pi) rather than (-pi/2, pi/2)
        self.radius = self.table[:, self.col_offset+self.n_properties_per_spot*np.arange(self.n_spots)]
        self.lat = self.table[:, self.col_offset+1+self.n_properties_per_spot*np.arange(self.n_spots)]
        self.lon = self.table[:, self.col_offset+2+self.n_properties_per_spot*np.arange(self.n_spots)]

        self.radius_col = self.col_offset+self.n_properties_per_spot*np.arange(self.n_spots)
        self.lat_col = self.col_offset+1+self.n_properties_per_spot*np.arange(self.n_spots)
        self.lon_col = self.col_offset+2+self.n_properties_per_spot*np.arange(self.n_spots)

        self.burnin_int = int(self.burnin*self.table.shape[0])

    def plot_chains(self):
        #fig, ax = plt.subplots(5)
        n_walkers = len((set(self.chain_ind)))
        fig, ax = plt.subplots(1, 3, figsize=(16, 8))

        for i in range(n_walkers):
            chain_i = self.chain_ind == i
            radius = self.table[chain_i, :][:, self.col_offset+self.n_properties_per_spot*np.arange(self.n_spots)]
            lat = self.table[chain_i, :][:, self.col_offset+1+self.n_properties_per_spot*np.arange(self.n_spots)]
            lon = self.table[chain_i, :][:, self.col_offset+2+self.n_properties_per_spot*np.arange(self.n_spots)]

            cmap = plt.cm.winter
            kwargs = dict(color=cmap(float(i)/n_walkers), alpha=0.3)
            ax[0].plot(radius, **kwargs)
            ax[1].plot(lat, **kwargs)
            ax[2].plot(lon, **kwargs)
            ax[0].set(title='Radius', xlabel='Step')
            ax[1].set(title='Latitude', xlabel='Step')
            ax[2].set(title='Longitude', xlabel='Step')
        fig.tight_layout()

        #plt.show()
        n_spots = self.radius.shape[1]
        fig, ax = plt.subplots(n_spots, 3, figsize=(16, 8))
        n_bins = 200
        for i in range(self.radius.shape[1]):
            ax[i, 0].hist(self.radius[self.burnin_int:, i], n_bins, color='k')#, alpha=0.3)
            ax[i, 1].hist(self.lat[self.burnin_int:, i], n_bins, log=True, color='k')#, alpha=0.3)
            ax[i, 2].hist(self.lon[self.burnin_int:, i], n_bins, log=True, color='k')#, alpha=0.3)
            ax[i, 0].set_ylabel('Spot {0}'.format(i))
        ax[0, 0].set(title='Radius')
        ax[0, 1].set(title='Latitude')
        ax[0, 2].set(title='Longitude')
        fig.tight_layout()

    def plot_each_spot(self):
        #fig, ax = plt.subplots(5)
        n_spots = self.radius.shape[1]
        burn_in_to_index = int(self.burnin*self.radius.shape[0])
        for i in range(n_spots):
            samples = np.array([self.radius[burn_in_to_index:, i],
                                self.lon[burn_in_to_index:, i]]).T # self.lat[:, i],
            corner.corner(samples)

            # cmap = plt.cm.winter
            # ax[0].plot(radius, color=cmap(float(i)/n_walkers))
            # ax[1].plot(lat, color=cmap(float(i)/n_walkers))
            # ax[2].plot(lon, color=cmap(float(i)/n_walkers))
            # ax[0].set(title='Radius', xlabel='Step')
            # ax[1].set(title='Latitude', xlabel='Step')
            # ax[2].set(title='Longitude', xlabel='Step')

    def plot_star(self, fade_out=True):
        spots_spherical = SphericalRepresentation(self.lon*u.rad,
                                                  (self.lat - np.pi/2)*u.rad,
                                                  1*R_sun)
        self.spots_spherical = spots_spherical
        fig, ax = plot_star(spots_spherical, fade_out=fade_out)
        #plt.show()

    def plot_corner(self):
        #fig = plt.figure(figsize=(14, 14))

        exclude_columns = np.array([0, 1, 2, 3, 5, 8, 11, 14, 17])
        include_columns = np.ones(self.table.shape[1])
        include_columns[exclude_columns] = 0
        fig = corner.corner(self.table[:, include_columns.astype(bool)])#, fig=fig)
        return fig

    def plot_chi2(self):
        n_walkers = len((set(self.chain_ind)))
        fig, ax = plt.subplots(figsize=(12, 12))
        n_spots = self.radius.shape[1]

        for i in range(n_walkers):
            chain_i = self.chain_ind == i

            chi2 = self.table[chain_i, 3]
            #ax.semilogx(range(len(chi2)), chi2)
            ax.semilogx(range(len(chi2)), np.log10(chi2), alpha=0.6)
            ax.set(xlabel='Step', ylabel=r'$\log_{10} \, \chi^2$')

def plot_star(spots_spherical, fade_out=False):
    """
    Parameters
    ----------
    spots_spherical : `~astropy.coordinates.SphericalRepresentation`
        Points in spherical coordinates that represent the positions of the
        star spots.
    """
    oldrcparams = matplotlib.rcParams
    matplotlib.rcParams['font.size'] = 18
    fig, ax = plt.subplots(2, 3, figsize=(16, 16))

    positive_x = ax[0, 0]
    negative_x = ax[1, 0]

    positive_y = ax[0, 1]
    negative_y = ax[1, 1]

    positive_z = ax[0, 2]
    negative_z = ax[1, 2]

    axes = [positive_z, positive_x, negative_z, negative_x, positive_y, negative_y]
    axes_labels = ['+z', '+x', '-z', '-x', '+y', '-y']

    # Set black background
    plot_props = dict(xlim=(-1, 1), ylim=(-1, 1), xticks=(), yticks=())
    drange = np.linspace(-1, 1, 100)
    y = np.sqrt(1 - drange**2)
    bg_color = 'k'
    for axis in axes:
        axis.set(xticks=(), yticks=())
        axis.fill_between(drange, y, 1, color=bg_color)
        axis.fill_between(drange, -1, -y, color=bg_color)
        axis.set(**plot_props)
        axis.set_aspect('equal')

    # Set labels
    positive_x.set(xlabel='$\hat{z}$', ylabel='$\hat{y}$') # title='+x',
    positive_x.xaxis.set_label_position("top")
    positive_x.yaxis.set_label_position("right")

    negative_x.set(xlabel='$\hat{z}$', ylabel='$\hat{y}$') # title='-x',
    negative_x.xaxis.set_label_position("top")

    positive_y.set(xlabel='$\hat{z}$', ylabel='$\hat{x}$')
    positive_y.xaxis.set_label_position("top")

    negative_y.set(xlabel='$\hat{z}$', ylabel='$\hat{x}$')
    negative_y.xaxis.set_label_position("top")
    negative_y.yaxis.set_label_position("right")

    negative_z.set(xlabel='$\hat{y}$', ylabel='$\hat{x}$') # title='-z',
    negative_z.yaxis.set_label_position("right")
    positive_z.set(xlabel='$\hat{y}$', ylabel='$\hat{x}$') # title='+z',
    positive_z.yaxis.set_label_position("right")
    positive_z.xaxis.set_label_position("top")

    for axis, label in zip(axes, axes_labels):
        axis.annotate(label, (-0.9, 0.9), color='w', fontsize=14,
                      ha='center', va='center')

    # Plot gridlines
    n_gridlines = 9
    print("lat grid spacing: {0} deg".format(180./(n_gridlines-1)))
    n_points = 35
    pi = np.pi

    latitude_lines = SphericalRepresentation(np.linspace(0, 2*pi, n_points)[:, np.newaxis]*u.rad,
                                             np.linspace(-pi/2, pi/2, n_gridlines).T*u.rad,
                                             np.ones((n_points, 1))
                                             ).to_cartesian()

    longitude_lines = SphericalRepresentation(np.linspace(0, 2*pi, n_gridlines)[:, np.newaxis]*u.rad,
                                              np.linspace(-pi/2, pi/2, n_points).T*u.rad,
                                              np.ones((n_gridlines, 1))
                                              ).to_cartesian()

    for i in range(latitude_lines.shape[1]):
        for axis in [positive_z, negative_z]:
            axis.plot(latitude_lines.x[:, i], latitude_lines.y[:, i],
                      ls=':', color='silver')
        for axis in [positive_x, negative_x, positive_y, negative_y]:
            axis.plot(latitude_lines.y[:, i], latitude_lines.z[:, i],
                      ls=':', color='silver')

    for i in range(longitude_lines.shape[0]):
        for axis in [positive_z, negative_z]:
            axis.plot(longitude_lines.y[i, :], longitude_lines.x[i, :],
                    ls=':', color='silver')
        for axis in [positive_x, negative_x, positive_y, negative_y]:
            axis.plot(longitude_lines.y[i, :], longitude_lines.z[i, :],
                ls=':', color='silver')


    # Plot spots
    spots_cart = spots_spherical.to_cartesian()
    spots_x = spots_cart.x/R_sun
    spots_y = spots_cart.y/R_sun
    spots_z = spots_cart.z/R_sun

    if fade_out:
        n = float(len(spots_x))
        alpha_range = np.arange(n)
        
        alpha = (n - alpha_range)/n
    else:
        alpha = 0.5

    for spot_ind in range(spots_x.shape[1]):

        above_x_plane = spots_x[:, spot_ind] > 0
        above_y_plane = spots_y[:, spot_ind] > 0
        above_z_plane = spots_z[:, spot_ind] > 0
        below_x_plane = spots_x[:, spot_ind] < 0
        below_y_plane = spots_y[:, spot_ind] < 0
        below_z_plane = spots_z[:, spot_ind] < 0

        positive_x.plot(spots_y[above_x_plane, spot_ind],
                        spots_z[above_x_plane, spot_ind], '.', alpha=alpha)

        negative_x.plot(-spots_y[below_x_plane, spot_ind],
                        spots_z[below_x_plane, spot_ind], '.', alpha=alpha)

        positive_y.plot(-spots_x[above_y_plane, spot_ind],
                        spots_z[above_y_plane, spot_ind], '.', alpha=alpha)

        negative_y.plot(spots_x[below_y_plane, spot_ind],
                        spots_z[below_y_plane, spot_ind], '.', alpha=alpha)

        positive_z.plot(spots_x[above_z_plane, spot_ind],
                        spots_y[above_z_plane, spot_ind], '.', alpha=alpha)

        negative_z.plot(spots_x[below_z_plane, spot_ind],
                        -spots_y[below_z_plane, spot_ind], '.', alpha=alpha)
    matplotlib.rcParams = oldrcparams
    return fig, ax


# blc = BestLightCurve(best_lc_path)
# blc.plot_whole_lc()
# blc.plot_transits()
# plt.show()

#mcmc = MCMCResults(mcmc_path)
#mcmc.plot_chains()
#mcmc.plot_star()
#plt.show()
