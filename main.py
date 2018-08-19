from job.drive import DriveJob
from job.dsa import DsaJob
import os

if __name__ == '__main__':
    EXPERIMENTS_DIR_PATH = "/Users/arvind.m.vepa/vessel_seg_data/experiments"
    EXPERIMENT_NAME = "drive_example_"
    job = DriveJob(OUTPUTS_DIR_PATH=os.path.join(EXPERIMENTS_DIR_PATH, EXPERIMENT_NAME))
    # keyword arguments are passed to several methods, please look at the code to see the flow of key-word arguments
    job.run_cross_validation(WRK_DIR_PATH="/Users/arvind.m.vepa/vessel_seg_data/drive",metrics_epoch_freq=None,
                             viz_layer_epoch_freq=1,n_epochs=5)
