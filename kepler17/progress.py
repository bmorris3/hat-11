import os
from glob import glob
import shutil
import numpy as np

run_name = 'kepler17'

top_level_output_dir = os.path.join('/astro/store/scratch/tmp/bmmorris/stsp',
                                    run_name)
                                    
#finished_runs = glob(os.path.join(top_level_output_dir, 
#                                  "window???/run???/*parambest.txt"))
                                  
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
    return completed_runs, runs_in_progress
    
completed_runs, runs_in_progress = find_windows_to_continue(top_level_output_dir)

import matplotlib.pyplot as plt

max_runs = dict()
for window, run in completed_runs:
    if window not in max_runs or max_runs[window] < run:
        max_runs[window] = run

window_number, run_number = zip(*max_runs.items())

fig, ax = plt.subplots(1, 2, figsize=(12, 6))
ax[0].plot(window_number, run_number, lw=2)#ls='steps', lw=2)
ax[0].set_xlabel('Window')
ax[0].set_ylabel('Run')
ax[0].axhline(np.mean(run_number))

ax[1].hist(run_number)
ax[1].set_xlabel('Latest completed runs')
plt.show()

