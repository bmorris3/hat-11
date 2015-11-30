import datetime
import sys
import subprocess
import os
from glob import glob
import shutil
import numpy as np

local_tmp_save_dir = "/local/tmp/bmmorris/stsp_tmp/"
stsp_executable = "/astro/users/bmmorris/git/STSP/bin/stsp"
stsp_executable_astrolab = "/astro/users/bmmorris/git/STSP/bin/stsp_astrolab"
run_name = 'kepler17'
top_level_output_dir = os.path.join('/astro/store/scratch/tmp/bmmorris/stsp',
                                    run_name)

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
            if os.path.exists(current_finalparam_path):
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

def begin_new_run(output_dir_path, window_index, run_index, job_id=None):
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
    print("Job ID: {0}".format(job_id))

    scratch_run_dir = os.path.join(output_dir_path,
                                    "window{0:03d}/run{1:03d}/"
                                    .format(window_index, run_index))

    initialized_path = os.path.join(scratch_run_dir, "initialized.txt"
                                    .format(window_index, run_index))

    local_window_dir = os.path.join(local_tmp_save_dir,
                                 'window{0:03d}'
                                 .format(window_index))

    local_run_dir = os.path.join(local_tmp_save_dir,
                                 'window{0:03d}/run{1:03d}'
                                 .format(window_index, run_index))

    if not os.path.exists(initialized_path):
        in_file = os.path.join(output_dir_path,
                               "window{0:03d}/run{1:03d}/window{0:03d}_run{1:03d}.in"
                               .format(window_index, run_index))
        dat_file = os.path.join(output_dir_path,
                                "window{0:03d}/window{0:03d}.dat"
                                .format(window_index))

        if not os.path.exists(local_run_dir):
            os.makedirs(local_run_dir)

        # Copy .in, .dat files
        shutil.copy(in_file, local_run_dir)
        shutil.copy(dat_file, local_run_dir)

        # If run is seeded, grab seed from previous run:
        if run_index != 0:
            seed_finalparam = os.path.join(output_dir_path,
                                          "window{0:03d}/run{1:03d}/window{0:03d}_run{1:03d}_finalparam.txt"
                                          .format(window_index, run_index-1))
            shutil.copy(seed_finalparam, local_run_dir)


        with open(initialized_path, 'w') as init:
            init.write('Initialized at {0}'.format(datetime.datetime.utcnow()))

        os.chdir(local_run_dir)
        p = subprocess.Popen([stsp_executable], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        o,e = p.communicate()
        if e=='':
            print('init job: ', stsp_executable, "window{0:03d}_run{1:03d}.in"
                             .format(window_index, run_index))
            subprocess.call([stsp_executable,
                             "window{0:03d}_run{1:03d}.in"
                             .format(window_index, run_index)],
                            cwd=local_run_dir)
        else:
            subprocess.call([stsp_executable_astrolab,
                             "window{0:03d}_run{1:03d}.in"
                             .format(window_index, run_index)],
                            cwd=local_run_dir)
        # copy data from /local/tmp dir back to shared dir
        for txt_file in glob(os.path.join(local_run_dir, "*.txt")):
            shutil.copy(txt_file, scratch_run_dir)

        # clean up after the script
        #os.system('rm /local/tmp/bmmorris/stsp_tmp//'+(sys.argv[1])[0:-3]+'*')
        shutil.rmtree(local_window_dir)

    else:
        with open(initialized_path, 'a') as init:
            init.write('Another initialization attempted at {0}'.format(datetime.datetime.utcnow()))

available_windows = find_windows_to_continue(top_level_output_dir)
if len(available_windows) < 1:
    raise ValueError("No available windows to run")

random_integer = np.random.randint(len(available_windows))
window_index, run_index = available_windows[random_integer]

print("Now beginning: (window, run) = ({0}, {1})".format(window_index, run_index))

begin_new_run(top_level_output_dir, window_index, run_index, sys.argv[-1])
