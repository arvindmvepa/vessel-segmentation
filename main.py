from job.drive import DriveJob
from job.dsa import DsaJob
import os

if __name__ == '__main__':
    EXPERIMENTS_DIR_PATH = "/Users/arvind.m.vepa/vessel_seg_data/experiments"
    EXPERIMENT_NAME = "drive_example"
    job = DriveJob(OUTPUTS_DIR_PATH=os.path.join(EXPERIMENTS_DIR_PATH, EXPERIMENT_NAME))
    job.run_single_model(WRK_DIR_PATH="/Users/arvind.m.vepa/vessel_seg_data/drive",metrics_epoch_freq=1,viz_layer_epoch_freq=1)
