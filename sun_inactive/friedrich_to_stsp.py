
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fmin

########################################################
# Import dev version of friedrich:
import sys
sys.path.insert(0, "/astro/users/bmmorris/git/friedrich/")
from friedrich.stsp import STSP, friedrich_results_to_stsp_inputs
from friedrich.lightcurve import hat11_params_morris
########################################################

friedrich_results_to_stsp_inputs('/local/tmp/friedrich/hat11', hat11_params_morris())

