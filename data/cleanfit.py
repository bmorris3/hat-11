"""
Tools for fitting a transit model to the clean "spotless" transits
generated in `clean_lightcurves.ipynb`

Fix period, ecc, w. Use params from fiducial least sq fit in
`datacleaner.TransitLightCurve.fiducial_transit_fit` to seed the run.


MCMC methods here have been adapted to allow input with either
quadratic or nonlinear (four parameter) limb-darkening parameterizations.
"""

import emcee
import numpy as np
import batman
import matplotlib.pyplot as plt
import astropy.units as u


def T14b2aRsi(P, T14, b):
    '''
    Convert from duration and impact param to a/Rs and inclination
    '''
    i = np.arccos( ( (P/np.pi)*np.sqrt(1 - b**2)/(T14*b) )**-1 )
    aRs = b/np.cos(i)
    return aRs, np.degrees(i)

def aRsi2T14b(P, aRs, i):
    '''
    Convert from a/Rs and inclination to duration and impact param
    '''
    b = aRs*np.cos(i)
    T14 = (P/np.pi)*np.sqrt(1-b**2)/aRs
    return T14, b

def generate_model_lc_short(times, t0, depth, dur, b, q1, q2, q3=None, q4=None):
    # LD parameters from Deming 2011 http://adsabs.harvard.edu/abs/2011ApJ...740...33D
    rp = depth**0.5
    exp_time = (1*u.min).to(u.day).value # Short cadence
    params = batman.TransitParams()
    params.t0 = t0                       #time of inferior conjunction
    params.per = 4.8878018                     #orbital period
    params.rp = rp                      #planet radius (in units of stellar radii)

    a, inc = T14b2aRsi(params.per, dur, b)

    params.a = a                       #semi-major axis (in units of stellar radii)
    params.inc = inc #orbital inclination (in degrees)
    params.ecc = 0                      #eccentricity
    params.w = 90.                       #longitude of periastron (in degrees)

    u1 = 2*np.sqrt(q1)*q2
    u2 = np.sqrt(q1)*(1 - 2*q2)

    if q3 is None and q4 is None:
        params.u = [u1, u2]                #limb darkening coefficients
        params.limb_dark = "quadratic"       #limb darkening model

    else:
        params.u = [q1, q2, q3, q4]
        params.limb_dark = "nonlinear"

    m = batman.TransitModel(params, times, supersample_factor=7,
                            exp_time=exp_time)
    model_flux = m.light_curve(params)
    return model_flux

def lnlike(theta, x, y, yerr):
    model = generate_model_lc_short(x, *theta)
    return -0.5*(np.sum((y-model)**2/yerr**2))

def lnprior(theta, bestfitt0=2454605.89132):
    if len(theta) == 6:
        t0, depth, dur, b, q1, q2 = theta
        if (0.001 < depth < 0.005 and 0.05 < dur < 0.15 and 0 < b < 1 and
            bestfitt0-0.1 < t0 < bestfitt0+0.1 and 0.0 < q1 < 1.0 and 0.0 < q2 < 1.0):
            return 0.0

    elif len(theta) == 8:
        t0, depth, dur, b, q1, q2, q3, q4 = theta
        if (0.001 < depth < 0.005 and 0.05 < dur < 0.15 and 0 < b < 1 and
            bestfitt0-0.1 < t0 < bestfitt0+0.1 and 0.0 < q1 < 1.0 and 0.0 < q2 < 1.0 and
            0.0 < q3 < 1.0 and 0.0 < q4 < 1.0):
            return 0.0
    return -np.inf

def lnprob(theta, x, y, yerr):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, x, y, yerr)

def run_emcee(p0, x, y, yerr, n_steps, n_threads=4, burnin=0.4):
    ndim = len(p0)
    nwalkers = 80
    n_steps = int(n_steps)
    burnin = int(burnin*n_steps)
    pos = [p0 + 1e-3*np.random.randn(ndim) for i in range(nwalkers)]

    pool = emcee.interruptible_pool.InterruptiblePool(processes=n_threads)
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(x, y, yerr),
                                    pool=pool)

    sampler.run_mcmc(pos, n_steps)
    samples = sampler.chain[:, burnin:, :].reshape((-1, ndim))
    return samples, sampler

def plot_triangle(samples):
    import triangle
    if samples.shape[1] == 6:
        fig = triangle.corner(samples, labels=["$t_0$", r"depth", r"duration",
                                               r"$b$", "$q_1$", "$q_2$"])
    elif samples.shape[1] == 8:
        fig = triangle.corner(samples, labels=["$t_0$", r"depth", r"duration",
                                               r"$b$", "$q_1$", "$q_2$", "$q_3$", "$q_4$"])
    plt.show()
