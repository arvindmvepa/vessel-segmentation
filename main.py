from job.drive import DriveJob
import os

if __name__ == '__main__':
    EXPERIMENTS_DIR_PATH = "/Users/arvind.m.vepa/vessel_seg_data/drive/experiments"
    EXPERIMENT_NAME = "example"
    job = DriveJob(OUTPUTS_DIR_PATH=os.path.join(EXPERIMENTS_DIR_PATH, EXPERIMENT_NAME))
    job.run_single_model(WRK_DIR_PATH="/Users/arvind.m.vepa/vessel_seg_data/drive",viz_layer_epoch_freq=1)