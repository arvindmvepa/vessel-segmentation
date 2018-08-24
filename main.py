from job.stare import StareJob
from job.drive import DriveJob
from job.hrf import HRFJob

import os

if __name__ == '__main__':
    experiments_dir_path = "/root/vessel-seg/data/STARE"
    experiment_name = "test"
    job = StareJob(OUTPUTS_DIR_PATH=os.path.join(experiments_dir_path, experiment_name))
    job.run_cross_validation(WRK_DIR_PATH="/root/vessel-seg/data/STARE",metrics_epoch_freq=1,
                             viz_layer_epoch_freq=1,n_epochs=5, gpu_device='/gpu:1' )
    print("done")

#    experiments_dir_path = "/root/vessel-seg/data/HRF"
#    experiment_name = "test"
#    job = HRFJob(OUTPUTS_DIR_PATH=os.path.join(experiments_dir_path, experiment_name))
#    job.run_cross_validation(WRK_DIR_PATH="/root/vessel-seg/data/HRF",metrics_epoch_freq=1,
#                         viz_layer_epoch_freq=1,n_epochs=5, gpu_device='/gpu:1')
#    print("done")
