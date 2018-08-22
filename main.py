from job.dsa import DsaJob
from job.drive import DriveJob
from job.stare import StareJob
from job.chase import ChaseJob
from imgaug import augmenters as iaa

import os

if __name__ == '__main__':

    ### OPTIMIZATION
    objective_fn = "wce"
    tuning_constant = 1.0
    ss_r = None
    regularizer_args = None
    learning_rate_and_kwargs = (.01, {"decay_epochs":10,"decay_rate":.1,"staircase":False})
    #learning_rate_and_kwargs = (.001, {})
    op_fun_and_kwargs = ("adam",{})

    ### LAYER ARGS
    weight_init = "He"
    act_fn = "lrelu"
    act_leak_prob = 0.10

    ### IMAGE PRE-PREPROCESSING
    hist_eq = True
    clahe_kwargs = {"clipLimit": 2.0,"tileGridSize":(8,8)}
    per_image_normalization = True
    gamma = 3.0

    # this image augmentation setting doesn't seem to be good
    """
    seq = iaa.Sequential([
        iaa.Crop(px=(0, 16)), # crop images from each side by 0 to 16px (randomly chosen)
        iaa.Fliplr(0.5), # horizontally flip 50% of the images
        iaa.GaussianBlur(sigma=(0, 3.0)) # blur images with a sigma of 0 to 3.0
        ])
    """
    seq = None

    ### OUTPUT INFO
    EXPERIMENTS_DIR_PATH = "/home/ubuntu/new_vessel_segmentation/vessel-segmentation/experiments"
    EXPERIMENT_NAME = "example"
    OUTPUTS_DIR_PATH = os.path.join(EXPERIMENTS_DIR_PATH, EXPERIMENT_NAME)
    metrics_epoch_freq = 1
    viz_layer_epoch_freq = 101
    n_epochs = 30

    ### JOB INFO
    WRK_DIR_PATH = "/home/ubuntu/new_vessel_segmentation/vessel-segmentation/drive"
    n_splits = 2

    job = DriveJob(OUTPUTS_DIR_PATH=OUTPUTS_DIR_PATH)
    job.run_cross_validation(WRK_DIR_PATH=WRK_DIR_PATH,
                             metrics_epoch_freq=metrics_epoch_freq,viz_layer_epoch_freq=viz_layer_epoch_freq,
                             n_epochs=n_epochs,n_splits=n_splits,objective_fn=objective_fn,
                             tuning_constant=tuning_constant, ss_r=ss_r,
                             regularizer_args=regularizer_args,
                             op_fun_and_kwargs=op_fun_and_kwargs,
                             learning_rate_and_kwargs=learning_rate_and_kwargs,
                             weight_init=weight_init, act_fn=act_fn, act_leak_prob=act_leak_prob,
                             seq=seq, hist_eq=hist_eq,
                             clahe_kwargs=clahe_kwargs,
                             per_image_normalization=per_image_normalization,
                             gamma=gamma)

