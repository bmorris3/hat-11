import datetime
import sys
import subprocess
import os
from glob import glob
import shutil
import numpy as np

local_tmp_save_dir = "/local/tmp/bmmorris/stsp_tmp/"
stsp_executable = "/home/bmorris/git/STSP/stsp_20160125"
#stsp_executable_astrolab = "/astro/users/bmmorris/git/STSP/bin/stsp_astrolab"
run_name = 'hat11'
#top_level_output_dir = os.path.join('/local-scratch/bmorris/hat11/',
#                                    run_name)
top_level_output_dir = os.path.join('/local-scratch/bmorris/hat11/',
                                    run_name)
os.chdir('/home/bmorris/git/hat-11/osg')
condor_template = open('template.osg-xsede', 'r').read()
condor_submit_path = 'condor_submit.osg-xsede'
falconer_log_path = 'falconer_log.txt'

def find_windows_to_continue(output_dir_path):
    """
    Parameters
    ----------
    output_dir_path : str
        Path to outputs

    Returns
    -------
    runs_ready_to_begin : list
        Runs that are ready to be begun
    """
    runs_ready_to_begin = []
    completed_runs = []
    runs_in_progress = []
    window_dirs = sorted(glob(os.path.join(output_dir_path, 'window*')))

    for window_dir in window_dirs:
        window_index = int(window_dir.split('window')[1])
        run_dirs = sorted(glob(os.path.join(window_dir, 'run*')))
        this_window_is_running_or_assigned = False

        for run_dir in run_dirs:
            run_index = int(run_dir.split('run')[1])

            run_id = (window_index, run_index)
            current_finalparam_path = os.path.join(run_dir, "window{0:03d}_run{1:03d}_finalparam.txt".format(window_index, run_index))
            current_initalized_path = os.path.join(run_dir, "initialized.txt")

            # If the current directory has its own finalparam.txt file, the run
            # has been completed
            if (os.path.exists(current_finalparam_path) and
                    os.path.getsize(current_finalparam_path) > 0):
                completed_runs.append(run_id)

            # If the current directory doesn't have its own finalparam.txt file,
            # but it has an initialized.txt file, the run is in progress
            elif os.path.exists(current_initalized_path):
                runs_in_progress.append(run_id)
                this_window_is_running_or_assigned = True

            # If run is not complete or in progress, it is uninitialized:
            elif not this_window_is_running_or_assigned:
                runs_ready_to_begin.append(run_id)
                this_window_is_running_or_assigned = True
    return runs_ready_to_begin

def begin_new_run(output_dir_path, window_index, run_index):
    """
    Parameters
    ----------
    output_dir_path : str
        Path to output directory
    window_ind : int
        Index of window for new job
    run_ind : int
        Index of run for new job
    """
    print(os.uname())

    run_dir = os.path.join(output_dir_path,
                                    "window{0:03d}/run{1:03d}/"
                                    .format(window_index, run_index))

    initialized_path = os.path.join(run_dir, "initialized.txt"
                                    .format(window_index, run_index))

    if not os.path.exists(initialized_path):
        with open(initialized_path, 'w') as init:
            init.write('Initialized at {0}'.format(datetime.datetime.utcnow()))

        in_file = os.path.join(output_dir_path,
                               "window{0:03d}/run{1:03d}/window{0:03d}_run{1:03d}.in"
                               .format(window_index, run_index))
        dat_file = os.path.join(output_dir_path,
                                "window{0:03d}/window{0:03d}.dat"
                                .format(window_index))

        # If run is seeded, grab seed from previous run:
        seed_finalparam_source = None
        if run_index != 0:
            seed_finalparam_source = os.path.join(output_dir_path,
                                          "window{0:03d}/run{1:03d}/window{0:03d}_run{1:03d}_finalparam.txt"
                                          .format(window_index, run_index-1))

            seed_finalparam_dest = os.path.join(output_dir_path,
                                                "window{0:03d}/run{1:03d}/"
                                                .format(window_index, run_index))

            shutil.copy(seed_finalparam_source, seed_finalparam_dest)



        output_files = ["window{0:03d}_run{1:03d}_{2}.txt".format(window_index, run_index, out)
                        for out in ["mcmc", "lcbest", "finalparam", "parambest"]]

        input_files = [i for i in [seed_finalparam_source, dat_file, in_file] if i is not None]

        condor_in = dict(xsede_allocation_name = 'TG-AST150046',
            initial_directory = run_dir,
            stsp_executable = '/home/bmorris/git/STSP/stsp_login',
            dot_in_file = in_file.split(os.sep)[-1],
            transfer_input_files = ", ".join(input_files),
            transfer_output_files = ", ".join(output_files),
            stdout_path = 'myout.txt',
            stderr_path = 'myerr.txt',
            log_path = 'mylog.txt')

        with open(condor_submit_path, 'w') as submit:
            submit.write(condor_template.format(**condor_in))

        with open(falconer_log_path, 'a') as fl:
            fl.write("Submitting: window{0:03d}_run{1:03d}\n".format(window_index, run_index))

        os.system('condor_submit {0}'.format(condor_submit_path))

    else:
        with open(initialized_path, 'a') as init:
            init.write('Another initialization attempted at {0}'.format(datetime.datetime.utcnow()))


available_windows = find_windows_to_continue(top_level_output_dir)
if len(available_windows) < 1:
    raise ValueError("No available windows to run")

random_integer = np.random.randint(len(available_windows))
#window_index, run_index = available_windows[0]#[random_integer]
#begin_new_run(top_level_output_dir, window_index, run_index, sys.argv[-1])

for available_window in available_windows:
    window_index, run_index = available_window
    begin_new_run(top_level_output_dir, window_index, run_index)
