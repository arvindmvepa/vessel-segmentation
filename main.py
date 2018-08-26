from job.dsa import DsaJob
from job.drive import DriveJob
from job.stare import StareJob
from job.chase import ChaseJob


import os

if __name__ == '__main__':
    experiments_dir_path = "/root/vessel-seg"
    experiment_name = "test"
    job = StareJob(OUTPUTS_DIR_PATH=os.path.join(experiments_dir_path, experiment_name))
    job.run_cross_validation(WRK_DIR_PATH="/root/vessel-seg/data/stare", metrics_epoch_freq=1, viz_layer_epoch_freq=1,
                             n_epochs=5)