from job.chase import ChaseJob
from job.stare import StareJob

import os

if __name__ == '__main__':
    experiments_dir_path = "/root/vessel-seg/data/CHASE"
    experiment_name = "test"
    job = ChaseJob(OUTPUTS_DIR_PATH=os.path.join(experiments_dir_path, experiment_name))
    job.run_single_model(WRK_DIR_PATH="/root/vessel-seg/data/CHASE")
    print("done")

"""
    experiments_dir_path = "/root/vessel-seg/data/STARE"
    experiment_name = "test"
    job = StareJob(OUTPUTS_DIR_PATH=os.path.join(experiments_dir_path, experiment_name))
    job.run_single_model(WRK_DIR_PATH="/root/vessel-seg/data/STARE")
    print("done")
"""
