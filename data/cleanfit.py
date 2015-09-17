"""
Tools for fitting a transit model to the clean "spotless" transits
generated in `clean_lightcurves.ipynb`

Fix period, ecc, w. Use params from fiducial least sq fit in
`datacleaner.TransitLightCurve.fiducial_transit_fit` to seed the run.

"""

import emcee
import numpy as np
import batman
import matplotlib.pyplot as plt
import astropy.units as u

def generate_model_lc_short(times, t0, depth, a, inc, u1, u2):
    # LD parameters from Deming 2011 http://adsabs.harvard.edu/abs/2011ApJ...740...33D
    rp = depth**0.5
    exp_time = (1*u.min).to(u.day).value # Short cadence
    params = batman.TransitParams()
    params.t0 = t0                       #time of inferior conjunction
    params.per = 4.8878018                     #orbital period
    params.rp = rp                      #planet radius (in units of stellar radii)
    params.a = a                       #semi-major axis (in units of stellar radii)
    params.inc = inc #orbital inclination (in degrees)
    params.ecc = 0                      #eccentricity
    params.w = 90.                       #longitude of periastron (in degrees)
    params.u = [u1, u2]                #limb darkening coefficients
    params.limb_dark = "quadratic"       #limb darkening model

    m = batman.TransitModel(params, times, supersample_factor=7,
                            exp_time=exp_time)
    model_flux = m.light_curve(params)
    return model_flux

def lnlike(theta, x, y, yerr):
    #t0, depth, a, inc, u1, u2 = theta
    model = generate_model_lc_short(x, *theta)
    return -0.5*(np.sum((y-model)**2/yerr**2))

def lnprior(theta, bestfitt0=2454605.89132):
    t0, depth, a, inc, u1, u2 = theta
    if (0.001 < depth < 0.005 and 85 < inc < 90 and 12 < a < 18 and
        bestfitt0-0.1 < t0 < bestfitt0+0.1 and 0.4 < u1 < 0.8 and 0.0 < u2 < 0.3):
        #0.5 < u1 < 0.7 and 0.0 < u2 < 0.2
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
    fig = triangle.corner(samples, labels=["$t_0$", r"$\delta$", r"$a/R_s$",
                                           r"$i$", "$u_1$", "$u_2$"])#,
                          #truths=[m_true, b_true, np.log(f_true)])
    #fig.savefig("triangle.png")
    plt.show()
