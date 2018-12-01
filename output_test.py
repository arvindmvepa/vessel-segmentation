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

    EXPERIMENTS_DIR_PATH = "/root/vessel-segmentation/experiments_output_test_eval"

    metrics_epoch_freq = 5
    viz_layer_epoch_freq = 10001

    WRK_DIR_PATH = "/root/vessel-segmentation/DRIVE"
    n_epochs = 200

    ### RANDOM SEARCH
    tuning_constant = 1.0
    objective_fn = "wce"
    regularizer_args = None

    op_fun_and_kwargs = ("adam", {})
    learning_rate_and_kwargs = (.001, {"decay_epochs": 50, "decay_rate": .1, "staircase": True})
    weight_init = "default"
    act_fn = "relu"

    hist_eq = False

    clahe_kwargs = None

    per_image_normalization = False
    gamma = 1.0

    seq = iaa.Sequential([iaa.Affine(rotate=(-45, 45), mode='constant', cval=0, name="rotate")])

    EXPERIMENT_NAME = get_experiment_string(seq, n_epochs)

    OUTPUTS_DIR_PATH = os.path.join(EXPERIMENTS_DIR_PATH, EXPERIMENT_NAME)

    job = DriveJob(OUTPUTS_DIR_PATH=OUTPUTS_DIR_PATH)
    job.run_single_model(WRK_DIR_PATH=WRK_DIR_PATH, early_stopping=False, early_stopping_metric="auc",
                         save_model=False, save_sample_test_images=False, metrics_epoch_freq=metrics_epoch_freq,
                         viz_layer_epoch_freq=viz_layer_epoch_freq, n_epochs=n_epochs, objective_fn=objective_fn,
                         tuning_constant=tuning_constant, regularizer_args=regularizer_args,
                         op_fun_and_kwargs=op_fun_and_kwargs, learning_rate_and_kwargs=learning_rate_and_kwargs,
                         weight_init=weight_init, act_fn=act_fn, seq=seq, hist_eq=hist_eq, clahe_kwargs=clahe_kwargs,
                         per_image_normalization=per_image_normalization,gamma=gamma)
