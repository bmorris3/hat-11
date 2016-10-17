import os
from glob import glob
import shutil
import numpy as np

run_name = 'kepler17'

top_level_output_dir = os.path.join('/astro/store/scratch/tmp/bmmorris/stsp',
                                    run_name)

from progress import find_windows_to_continue

completed_runs, runs_in_progress = find_windows_to_continue(top_level_output_dir)

print("Runs to remove inits from:", runs_in_progress)

message = "Are you sure you want to remove all dangling init files? Enter 'yes': "
if not raw_input(message).strip().lower() == 'yes':
    raise ValueError('Killing `{0}`.'.format(__file__))


for window, run in runs_in_progress:
    init_path = os.path.join(top_level_output_dir,
                             "window{0:03d}".format(window),
                             "run{0:03d}".format(run),
                             "initialized.txt")
    os.remove(init_path)

print("Init files removed.")