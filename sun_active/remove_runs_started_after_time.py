import os
from glob import glob
import shutil
import numpy as np
import datetime

run_name = 'kepler17'

top_level_output_dir = os.path.join('/astro/store/scratch/tmp/bmmorris/stsp',
                                    run_name)

from progress import find_windows_to_continue

completed_runs, runs_in_progress = find_windows_to_continue(top_level_output_dir)

remove_runs_after_this_time = datetime.datetime(2015, 12, 1, 13)

message = "Are you sure you want to remove all files created after {0}? Enter 'yes': ".format(remove_runs_after_this_time)
if not raw_input(message).strip().lower() == 'yes':
    raise ValueError('Killing `{0}`.'.format(__file__))

for completed_run in completed_runs:
    window_ind, run_ind = completed_run
    run_dir = os.path.join(top_level_output_dir, 'window{0:03d}'.format(window_ind),
                                   'run{0:03d}'.format(run_ind))
    finalparam_path = os.path.join(run_dir,
                                   "window{0:03d}_run{1:03d}_finalparam.txt".format(window_ind, run_ind))
    made_after_cutoff = datetime.datetime.fromtimestamp(os.path.getctime(finalparam_path)) > remove_runs_after_this_time
    #print(finalparam_path, made_after_cutoff)
    if made_after_cutoff:
        txt_files = glob(os.path.join(run_dir,'*.txt'))
        for txt_file in txt_files:
            os.remove(txt_file)
