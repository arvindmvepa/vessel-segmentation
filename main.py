from job.drive import DriveJob
from job.dsa import DsaJob
import os

if __name__ == '__main__':
    EXPERIMENTS_DIR_PATH = "c:/vessel-segmentation/experiments"
    EXPERIMENT_NAME = "/drive_example_/"
    job = DriveJob(OUTPUTS_DIR_PATH=os.path.join(EXPERIMENTS_DIR_PATH, EXPERIMENT_NAME))
    job.run_cross_validation(WRK_DIR_PATH="c:/vessel-segmentation/DRIVE",metrics_epoch_freq=1,
            viz_layer_epoch_freq=1,n_epochs=5,gpu_device='/gpu:1', weight_init='He',regularizer='L2',Relu=False,
            learningrate=0.001, Beta1=0.9,Beta2=0.999,epsilon=10**-8)
