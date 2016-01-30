# -*- coding: utf-8 -*-

import numpy as np
import os
import matplotlib.pyplot as plt

def smooth(x, window_len=11, window='hanning'):
    """smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    input:
        x: the input signal
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal

    example:
        t=linspace(-2,2,0.1)
        x=sin(t)+randn(len(t))*0.1
        y=smooth(x)

    see also:
        numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
        scipy.signal.lfilter
    """
    if x.ndim != 1:
        raise ValueError, "smooth only accepts 1 dimension arrays."
    if x.size < window_len:
        raise ValueError, "Input vector needs to be bigger than window size."
    if window_len<3:
        return x
    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"

    s=np.r_[x[window_len-1:0:-1],x,x[-1:-window_len:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    y=np.convolve(w/w.sum(), s, mode='valid')
    return y[(window_len/2-1):-(window_len/2)]

#############
#    SET UP PARAMETERS

#prefix = 'kepler17'
#BJDREF = 2454833.
# prefix = 'joe'
# BJDREF = 0.

prefix = 'kep17flat'
BJDREF = 2454833.


# action can be = 'M', 'L', 'T', 'S'
action = 'M'


# where the results are stored
# !!- Make sure to include / at the end -!!
workdir = '/astro/store/scratch/jrad/stsp/'+prefix+'/'
dropboxdir = '/astro/users/jrad/Dropbox/python/kepler17/flat_lc/'


# do seeded runs based on previous runs?
USESEED = False
seeddir = '/astro/store/scratch/jrad/stsp/notyet/'
seedprefix = 'notyet'


# do flattened mode? (also flatten LC, toss out-of-transit data)
FLATMODE = True


# 1111111111111111
#  SET UP MCMC PARAMETERS
#nspot_list = ['5','6','7']
nspot_list = ['8']  # spots to model, code will loop over this list at each timestep

ascale = '2.5' # jump parameter
nsteps = '1000' # MCMC steps
npop = '500' # walkers

# where is the original data file to analyze?
datafile = 'kepler17_whole.dat'
tt0,ff0,ee0 = np.loadtxt(dropboxdir + datafile,unpack=True)

#-- for dealing w/ Joe's model data:
# datafile = dropboxdir + 'llama_lightcurve_raw.txt'
# tbad,ff0 = np.loadtxt(datafile, unpack=True, usecols=(0,1), dtype=('string','float'))
# escale = 0.0002 # value i arrived at "a posteriori"
# ee0 = np.random.normal(loc=0,scale=escale,size=np.size(ff0))
# # remake time array
# dt = 5. / 60. / 24. # 5 min cadence
# tt0 = np.arange(np.size(ff0))*dt


# the 4th order limb darkening array
limb_array = '0.59984  -0.165775  0.6876732  -0.349944' # for Kep 17
# limb_array = '0.7072, -0.7489, 1.4971, -0.6442' # for Joe's model

# the files to create...
shellscript = dropboxdir + 'pylaunch_'+prefix+'.csh'
cfgfile     = dropboxdir + 'condor_'+prefix+'.condor'
stsprunner  = dropboxdir + 'stsprun_'+prefix+'.py'

# the version of the STSP code.
stsp_ver = '/astro/users/jrad/research/kepler/STSP/bin/stsp'

#pyversion = "/usr/bin/python"
pyversion = "/astro/apps6/anaconda/bin/python"

### what to call each STSP .in file
infilename = prefix+'.in'



# the parameter file detailing the system
pfile = 'kep17.params'

params = np.loadtxt(dropboxdir + pfile,unpack=True,skiprows=4,usecols=[0])
p_orb = str(params[0])
t_0 = str(params[1] - BJDREF)
tdur = str(params[2])
p_rot = str(params[3])
Rp_Rs = str(params[4]**2.)
impact = str(params[5])
incl_orb = str(params[6])
Teff = str(params[7])
sden = str(params[8])

############# END OF PARAMETER SETUP

toff = float(tdur) / float(p_orb) # the transit duration in % (phase) units

# down-sample the out-of-transit data
# define the region +/- 1 transit duration for in-transit
OOT = np.where( ((((tt0-float(t_0)) % float(p_orb))/float(p_orb)) > toff) &
                ((((tt0-float(t_0)) % float(p_orb))/float(p_orb)) < 1-toff) )

ITT = np.where( ((((tt0-float(t_0)) % float(p_orb))/float(p_orb)) <= toff) |
                ((((tt0-float(t_0)) % float(p_orb))/float(p_orb)) >= 1-toff) )


# the factor to down-sample by
Nds = 30 # for Kepler 17
# Nds = 10 # for Joe's data

# down sample out-of-transit, grab every N'th data point
OOT_ds = OOT[0][np.arange(0,np.size(OOT), Nds)]

if FLATMODE is True:
    # just use in-transit data
    idx = ITT[0][0:]
    idx.sort()
elif FLATMODE is False:
    # use in- and out-of-transit data
    idx = np.concatenate((ITT[0][0:], OOT_ds))
    idx.sort()

# these arrays are the final, down-sampled data
tt = tt0[idx]
ff = ff0[idx]
ee = ee0[idx]
#-- for Joe's model, add noise
# ff = np.add( np.array(ff0[idx], dtype='float'), ee0[idx] )
# ee = np.ones_like(ff) * escale


medflux = np.median(ff)


# if using FLATMODE, loop over every transit
# grab before & after transit, flatten
if FLATMODE is True:
    # empty arrays
    tt1 = np.array([],dtype='float')
    ff1 = np.array([],dtype='float')
    ee1 = np.array([],dtype='float')

    ff0_smooth = smooth(ff0, window_len=11)
    ff_max = np.max(ff0_smooth)

    for j in range(int( (max(tt)-min(tt))/float(p_orb)+1 )):
        # the center time of the j'th transit
        trans_j = float(t_0) + float(p_orb)*j

        # define the out-of-transit regions around the transit
        c1 = np.where((tt >= trans_j - float(tdur)*1.0) & # Left side of window
                      (tt <= trans_j - float(tdur)*0.6))  # Left inner side
        c2 = np.where((tt >= trans_j + float(tdur)*0.6) & # Right inner side
                      (tt <= trans_j + float(tdur)*1.0))  # Right side of window

        # the whole region around the event
        c = np.where((tt >= trans_j - float(tdur)*1.1) & # Left side of window
                     (tt <= trans_j + float(tdur)*1.1))  # Right side of window

        if (len(c1[0])>0) and (len(c2[0])>0):
            c12 = np.concatenate((c1[0], c2[0]))
            fit = np.polyfit(tt[c12], ff[c12], 1)
            line = np.polyval(fit, tt[c])

            tt1 = np.concatenate((tt1, tt[c]))
            ee1 = np.concatenate((ee1, ee[c]))
            ff1 = np.concatenate((ff1, ff[c] - line + ff_max))

    # print(tt0.shape)
    # plt.figure()
    # plt.plot(tt0,ff0, '.')
    # plt.show()

    tt = tt1
    ff = ff1
    ee = ee1

# 2222222222222222 create CONDOR .cfg file
f2 = open(cfgfile,'w')
f2.write('Notification = never \n')
f2.write('Executable = '+shellscript+' \n')
f2.write('Initialdir = ' + workdir + '\n')
f2.write('Universe = vanilla \n')
f2.write(' \n')
f2.write('Log = '+workdir+'log.txt \n')
f2.write('Error = '+workdir+'err.txt \n')
f2.write('Output = '+workdir+'out.txt \n')
f2.write(' \n')



## use entire range in file
# dstart = np.min(tt)
# ddur = np.max(tt)-np.min(tt)

# ntrials = len(nspots)
# rk = range(ntrials)


# In each time window use slightly less than p_rot to avoid
# problems at edge of window due to rapid spot evolution
ddur = float(p_rot) * 0.9
# ddur = float(p_rot) * 1.2

# of time windows, shift each window by 1/2 window size
ntrials = np.floor((np.max(tt)-np.min(tt))*2.0/float(p_rot))
nspots = np.tile(nspot_list, ntrials)

dstart_all = np.repeat(np.min(tt) + float(p_rot)/2.*np.arange(ntrials), len(nspot_list))


# main loop for .in file writing of each time window & #spots
for n in range(len(dstart_all)):
    dstart = dstart_all[n]
    npts = np.sum((tt >= dstart) & (tt <= dstart+ddur))
    wndw = np.where((tt >= dstart) & (tt <= dstart+ddur)) # this could be combined w/ above...

    # define window number
    namen = format(int(n+1),"03d")

#    print namen, npts, np.size(wndw), npts>=100, os.path.isfile(seeddir + namen + seedprefix+'_parambest.txt')

    # only run STSP on line if more than N epoch of data
    if (USESEED and npts >= 100 and
            os.path.isfile(seeddir + namen + seedprefix+'_parambest.txt')) \
            or (not USESEED):
        # write small chunk of LC to operate on
        datafile_n = workdir + namen + prefix+ '.dat'
        dfn = open(datafile_n, 'w')
        for k in range(0,npts-1):
            dfn.write(str(tt[wndw[0][k]]) + ' ' +
                      str(ff[wndw[0][k]]) + ' ' +
                      str(ee[wndw[0][k]]) + '\n')
        dfn.close()

        # fit this chunk w/ fourier model, find max
        #==> need to figure out

        # name of .in file to use for this time window:
        file = workdir + namen + infilename
        f = open(file, 'w')

        f.write('#PLANET PROPERTIES\n')
        f.write('1\n')            #-- # planets
        f.write(t_0 + '\n')       #-- T0  (middle of first transit) in days.
        f.write(p_orb + '\n')     #-- Planet Period (days)
        f.write(Rp_Rs + '\n')     #-- (Rp/Rs)^2
        f.write(tdur + '\n')      #-- Duration (days)
        f.write(impact + '\n')    #-- Impact parameter (0=planet cross equator)
        f.write(incl_orb + '\n')  #-- Incl angle of orbit (90deg=planet cross over equator)
        f.write('0\n' )           #-- sky projected obliquity ("lambda")
        f.write('0.0\n')          #-- ecosw (leave 0)
        f.write('0.0\n')          #-- esinw (leave 0)

        f.write('#STAR PROPERTIES\n')
        f.write(sden + '\n')          #-- stellar density ( this from Leslie )
        f.write(p_rot + '\n')          #-- rotation period
        f.write(Teff + '\n')           #-- Teff
        f.write('0.0\n')               #-- Metallicity (not used)
        f.write('0.0\n')               #-- inclination
        f.write(limb_array + '\n') #-- limb darkening (from Leslie's calculation)
        f.write('100\n')               #-- # rings for limb darkening approx.

        f.write('#SPOT PROPERTIES\n')
        f.write(nspots[n] +'\n')       #-- # of spots
        f.write('0.7\n')               #-- contrast

        f.write('# FITTING properties\n')
        f.write(datafile_n + '\n')     #-- the datafile
        f.write(str(dstart) + '\n')    #-- tstart
        f.write(str(ddur) + '\n')      #-- duration (days)


        # use smoothed LC max
        if FLATMODE is True:
            # ff_smooth = smooth(ff[wndw], window_len=11)
            # maxwindow = np.max(ff_smooth)
            maxwindow = ff_max
        else:
            maxwindow = medflux

        f.write(str(maxwindow)+'\n')   #-- real max of LC, 0=use downfrommax

        if FLATMODE is True:
            f.write('1\n')
        if FLATMODE is False:
            f.write('0\n')                 #-- is LC flattened to zero outside of transit? (1=yes,0=no)

        f.write('#ACTION\n')           #   what to do w/ STSP?
        if not USESEED:
            f.write('M\n')                 #-- full MCMC search
        if USESEED:
            f.write('S\n')                 #-- seeded w/ parameters from file


        f.write(str(int(np.abs(np.random.rand()*1e2))) + '\n') #-- random seed

        f.write(ascale + '\n')         #-- mcmc ascale
        f.write(npop + '\n')           #-- npop (# chains or walkers)
        f.write(nsteps + '\n')         #-- # mcmc steps (chain length)

        #not used currently (v4.4)
        #f.write('1\n')                #-- combine 1 spot at a time (0=all at once)

        f.write('1.0' + '\n')          #-- 1=calculate the brightness correction,0=downfrommax only

        if USESEED:
            f.write('0.01' + '\n')      # sigma for radius variation
            f.write('0.02' + '\n')      # sigma for angle variation
            f.write(seeddir + namen + seedprefix+'_parambest.txt') # seed input (parambest) file
        f.close()


        # put entry in to CONDOR .cfg file for this window
        f2.write('Arguments = ' + namen + infilename + ' \n')
        f2.write('Queue \n')
    
f2.write(' \n')
f2.close()



# 33333333333333333
# create the very simple PYTHON-launching shell script
f3 = open(shellscript,'w')
f3.write("#!/bin/bash \n")
f3.write(pyversion + " " + stsprunner + " $1 \n")
f3.close()




# 444444444444444  generate the STSP-running script
f4 = open(stsprunner,'w')

f4.write("import sys \n")
f4.write("import os \n")
f4.write("import subprocess \n")
f4.write("workpath = '"+workdir+"' \n")
f4.write("stsp_ver = '"+stsp_ver+"' \n")
f4.write("if not os.path.isdir('/local/tmp/jrad/stsp_tmp'): \n")
f4.write("    try:  \n")
f4.write("        os.makedirs('/local/tmp/jrad/stsp_tmp') \n")
f4.write("    except OSError: \n")
f4.write("        pass \n")
f4.write("\n")
#f4.write("os.system('find /local/tmp/jrad/stsp_tmp/* -mmin +240 -delete')\n")
f4.write("os.system('cp '+workpath+sys.argv[1]+' /local/tmp/jrad/stsp_tmp/.') \n")
f4.write("p = subprocess.Popen([stsp_ver], stdout=subprocess.PIPE, stderr=subprocess.PIPE) \n")
f4.write("o,e = p.communicate() \n")
# have GLIBC_2.7 (grad machines)
f4.write("if e=='': \n")
f4.write("    subprocess.call([stsp_ver, '/local/tmp/jrad/stsp_tmp/'+sys.argv[1] ]) \n")
# otherwise (astrolabs, older machines)
f4.write("if not e=='': \n")
f4.write("    subprocess.call([stsp_ver+'a', '/local/tmp/jrad/stsp_tmp/'+sys.argv[1] ]) \n")
# copy data from /local/tmp dir back to shared dir
f4.write("os.system('cp /local/tmp/jrad/stsp_tmp/'+(sys.argv[1])[0:-3]+'* '+workpath+'.') \n")
# clean up after the script
f4.write("os.system('rm /local/tmp/jrad/stsp_tmp/'+(sys.argv[1])[0:-3]+'*') \n")
##f4.write("os.system('find /local/tmp/jrad/stsp_tmp/. -mtime +2 -delete')\n")
f4.write("print sys.argv\n")
f4.write("print os.uname()\n")
f4.write(" ")
f4.close()


# fix permissions
os.system("chmod 777 "+stsprunner)
os.system("chmod 777 "+shellscript)
os.system("chmod 777 "+cfgfile)

# how to launch CONDOR jobs!
# actually, no... run THIS file on "ame", and launch condor jobs on "albatross"
### os.system("condor_submit "+cfgfile)
