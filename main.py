from job.dsa import DsaJob
from job.drive import DriveJob
from job.stare import StareJob
from job.chase import ChaseJob
from imgaug import augmenters as iaa
from itertools import product
from random import sample
import os

if __name__ == '__main__':

    ### OPTIMIZATION
    op_fun_and_kwargs = ("adam", {})
    tuning_constant = 1.0
    ss_r = .5
    objective_fn = "wce"
    regularizer_args = None
    learning_rate_and_kwargs = (.01, {})

    ### LAYER ARGS
    weight_init = "He"
    act_fn = "lrelu"
    act_leak_prob = 0.2

    ### IMAGE PRE-PREPROCESSING
    hist_eq = False
    clahe_kwargs = None
    per_image_normalization = False
    gamma = 1.0

    ### IMAGE AUGMENTATION
    # this image augmentation setting doesn't seem to be good
    """
    seq = iaa.Sequential([
        iaa.Crop(px=(0, 16)), # crop images from each side by 0 to 16px (randomly chosen)
        iaa.Fliplr(0.5), # horizontally flip 50% of the images
        iaa.GaussianBlur(sigma=(0, 3.0)) # blur images with a sigma of 0 to 3.0
        ])
    """
    seq = None

    ### JOB INFO
    Job_cls = DsaJob
    WRK_DIR_PATH = "/home/ubuntu/new_vessel_segmentation/vessel-segmentation/dsa"
    n_splits = 3

    ### OUTPUT INFO
    EXPERIMENTS_DIR_PATH = "/home/ubuntu/new_vessel_segmentation/vessel-segmentation/experiments2"
    EXPERIMENT_NAME = str((objective_fn,tuning_constant,ss_r if objective_fn=="ss" else None,regularizer_args,op_fun_and_kwargs,
                           learning_rate_and_kwargs, weight_init, act_fn, act_leak_prob, False if seq is None else True,
                           hist_eq, clahe_kwargs, per_image_normalization,gamma))
    OUTPUTS_DIR_PATH = os.path.join(EXPERIMENTS_DIR_PATH, EXPERIMENT_NAME)
    metrics_epoch_freq = 1
    viz_layer_epoch_freq = 101
    n_epochs = 5

    job = Job_cls(OUTPUTS_DIR_PATH=OUTPUTS_DIR_PATH)
    job.run_cv(WRK_DIR_PATH=WRK_DIR_PATH, mc=True, early_stopping=True, early_stopping_metric="auc",
               save_model=False, save_sample_test_images=False,
               metrics_epoch_freq=metrics_epoch_freq, viz_layer_epoch_freq=viz_layer_epoch_freq,
               n_epochs=n_epochs, n_splits=n_splits, objective_fn=objective_fn,
               tuning_constant=tuning_constant, ss_r=ss_r,
               regularizer_args=regularizer_args,
               op_fun_and_kwargs=op_fun_and_kwargs,
               learning_rate_and_kwargs=learning_rate_and_kwargs,
               weight_init=weight_init, act_fn=act_fn, act_leak_prob=act_leak_prob,
               seq=seq, hist_eq=hist_eq,
               clahe_kwargs=clahe_kwargs,
               per_image_normalization=per_image_normalization,
               gamma=gamma)

