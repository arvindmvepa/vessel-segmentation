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
    ss_r = None
    objective_fn = "wce"
    regularizer_args = None
    learning_rate_and_kwargs = (.001, {})

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
    seq = None

    ### JOB INFO
    Job_cls = DriveJob
    WRK_DIR_PATH = "C:\\Users\\arvin\\dev\\vessel-segmentation\\DRIVE"
    n_splits = 2

    ### OUTPUT INFO
    EXPERIMENTS_DIR_PATH = "C:\\Users\\arvin\\dev\\vessel-segmentation\\arch_tests"

    EXPERIMENT_NAME = "test"
    OUTPUTS_DIR_PATH = os.path.join(EXPERIMENTS_DIR_PATH, EXPERIMENT_NAME)
    metrics_epoch_freq = 1
    viz_layer_epoch_freq = 1
    n_epochs = 5

    job = Job_cls(OUTPUTS_DIR_PATH=OUTPUTS_DIR_PATH)
    job.run_cv(WRK_DIR_PATH=WRK_DIR_PATH, mc=True, val_prop=.1, save_model=False, save_sample_test_images=False,
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

