from job.dsa import DsaJob
from job.drive import DriveJob
from job.stare import StareJob
from job.chase import ChaseJob

import os

if __name__ == '__main__':
    EXPERIMENTS_DIR_PATH = "/home/ubuntu/new_vessel_segmentation/vessel-segmentation/experiments"
    EXPERIMENT_NAME = "drive_example"
    job = DriveJob(OUTPUTS_DIR_PATH=os.path.join(EXPERIMENTS_DIR_PATH, EXPERIMENT_NAME))
    # keyword arguments are passed to several methods, please look at the code to see the flow of key-word arguments
    job.run_cross_validation(WRK_DIR_PATH="/home/ubuntu/new_vessel_segmentation/vessel-segmentation/drive",metrics_epoch_freq=1,
                             viz_layer_epoch_freq=1,n_epochs=5)
