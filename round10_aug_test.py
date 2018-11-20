from job.drive import DriveJob
from itertools import product
from random import sample
import os
import json
from copy import deepcopy
from imgaug import augmenters as iaa
from imgaug import parameters as iap

def get_experiment_string(seq, n_epochs, sep="__"):

    exp_string = ""
    if seq is not None:
        for aug in list(seq):
            exp_string += aug.name
        exp_string += sep
    else:
        exp_string += str(seq) + sep

    exp_string += str(n_epochs)

    exp_string = exp_string.replace("\'","").replace("\"","").replace(",","").replace(" ","-").\
        replace("{","(").replace("}",")").replace("[","(").replace("]",")").replace(":","-")

    #exp_string = exp_string.replace("\\", "").replace('/', '')
    return exp_string

if __name__ == '__main__':

    ### DEBUG THIS TO CHECK EPOCH COUNTS
    ### WAS A BUG WHEN CALCULATING MEAN OF SCORE SHEETS
    ### AT MINIMUM SHOULD CHANGE SCORING FOR MORE ENTRIES (THOUGH THAT SHOULD BE FOR THE RESULST ANALYZER, NOT SURE ISSUE)

    EXPERIMENTS_DIR_PATH = "/root/vessel-segmentation/experiments10_aug_test_eval"
    # EXPERIMENTS_DIR_PATH = "C:\\vessel-segmentation\\aug_exp1"

    metrics_epoch_freq = 5
    viz_layer_epoch_freq = 10001

    WRK_DIR_PATH = "/root/vessel-segmentation/DRIVE"
    #WRK_DIR_PATH = "C:\\vessel-segmentation\\drive"
    n_epochs = 200

    tuning_constant = 1.5
    objective_fn = "wce"
    regularizer_args = ("L1",1E-8)

    op_fun_and_kwargs = ("rmsprop", {})
    learning_rate_and_kwargs = (.001, {"decay_epochs": 50, "decay_rate": .1, "staircase": True})
    weight_init = "default"
    act_fn = "lrelu"
    act_leak_prob = 0.2

    hist_eq = False

    clahe_kwargs = {"clipLimit": 2.0,"tileGridSize":(8,8)}

    per_image_normalization = True
    gamma = 2.0

    seqs = [iaa.Sequential([iaa.Affine(rotate=(-45, 45), mode='constant', cval=0, name="rotate")]),
            iaa.Sequential([iaa.Affine(rotate=(-135, 135), mode='constant', cval=0, name="rotate")]),
            iaa.Sequential([iaa.Affine(rotate=(-180, 180), mode='constant', cval=0, name="rotate")]),
            ]

    for seq in seqs:

        EXPERIMENT_NAME = get_experiment_string(seq, n_epochs)

        OUTPUTS_DIR_PATH = os.path.join(EXPERIMENTS_DIR_PATH, EXPERIMENT_NAME)

        job = DriveJob(OUTPUTS_DIR_PATH=OUTPUTS_DIR_PATH)
        job.run_single_model(WRK_DIR_PATH=WRK_DIR_PATH, early_stopping=False, early_stopping_metric="auc",
                             save_model=False, save_sample_test_images=False, metrics_epoch_freq=metrics_epoch_freq,
                             viz_layer_epoch_freq=viz_layer_epoch_freq, n_epochs=n_epochs, objective_fn=objective_fn,
                             tuning_constant=tuning_constant, regularizer_args=regularizer_args,
                             op_fun_and_kwargs=op_fun_and_kwargs, learning_rate_and_kwargs=learning_rate_and_kwargs,
                             weight_init=weight_init, act_fn=act_fn, act_leak_prob=act_leak_prob, seq=seq,
                             hist_eq=hist_eq, clahe_kwargs=clahe_kwargs,
                             per_image_normalization=per_image_normalization,gamma=gamma)

    n_epochs = 100
    learning_rate_and_kwargs = (.001, {"decay_epochs": 25, "decay_rate": .1, "staircase": True})

    for seq in seqs:

        EXPERIMENT_NAME = get_experiment_string(seq, n_epochs)
        OUTPUTS_DIR_PATH = os.path.join(EXPERIMENTS_DIR_PATH, EXPERIMENT_NAME)
        job = DriveJob(OUTPUTS_DIR_PATH=OUTPUTS_DIR_PATH)

        job.run_single_model(WRK_DIR_PATH=WRK_DIR_PATH, early_stopping=False, early_stopping_metric="auc",
                             save_model=False, save_sample_test_images=False, metrics_epoch_freq=metrics_epoch_freq,
                             viz_layer_epoch_freq=viz_layer_epoch_freq, n_epochs=n_epochs, objective_fn=objective_fn,
                             tuning_constant=tuning_constant, regularizer_args=regularizer_args,
                             op_fun_and_kwargs=op_fun_and_kwargs, learning_rate_and_kwargs=learning_rate_and_kwargs,
                             weight_init=weight_init, act_fn=act_fn, act_leak_prob=act_leak_prob, seq=seq,
                             hist_eq=hist_eq, clahe_kwargs=clahe_kwargs, per_image_normalization=per_image_normalization,
                             gamma=gamma)
