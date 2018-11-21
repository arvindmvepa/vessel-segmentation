from job.drive import DriveJob
from itertools import product
from random import sample
import os
import json
from copy import deepcopy
from imgaug import augmenters as iaa
from imgaug import parameters as iap

def get_experiment_string(**kwargs):
    exp_string = ""
    if "objective_fn" in kwargs:
        exp_string += "objective_fn"+kwargs["objective_fn"]
    if "tuning_constant" in kwargs:
        exp_string += "tuning_constant" + (str(kwargs["tuning_constant"]))
    if "ss_r" in kwargs:
        exp_string += "ss_r" + str(kwargs["ss_r"])
    if "regularizer_args" in kwargs:
        exp_string += "regularizer_args" + str(kwargs["regularizer_args"])
    if "op_fun_and_kwargs" in kwargs:
        exp_string += "op_fun_and_kwargs" + str(kwargs["op_fun_and_kwargs"])
    if "learning_rate_and_kwargs" in kwargs:
        learning_rate_and_kwargs = deepcopy(kwargs["learning_rate_and_kwargs"])
        learning_rate_kwargs = learning_rate_and_kwargs[1]
        if "decay_epochs" in learning_rate_kwargs:
            learning_rate_kwargs["d_e"]=learning_rate_kwargs.pop("decay_epochs")
        if "decay_rate" in learning_rate_kwargs:
            learning_rate_kwargs["d_r"] = learning_rate_kwargs.pop("decay_rate")
        if "staircase" in learning_rate_kwargs:
            learning_rate_kwargs["s_c"] = learning_rate_kwargs.pop("staircase")
        exp_string += "learning_rate_and_kwargs" + "("+str(learning_rate_and_kwargs[0]) +","+json.dumps(learning_rate_kwargs, sort_keys=True) + ")"
    if "weight_init" in kwargs:
        exp_string += "weight_init"+kwargs["weight_init"]
    if "act_fn" in kwargs:
        exp_string += "act_fn"+kwargs["act_fn"]
    if "act_leak_prob" in kwargs:
        exp_string += "act_leak_prob"+str(kwargs["act_leak_prob"])
    if "hist_eq" in kwargs:
        exp_string += "hist_eq"+str(kwargs["hist_eq"])
    if "seq" in kwargs:
        seq = kwargs["seq"]
        if seq is not None:
            for aug in list(seq):
                exp_string += aug.name
        else:
            exp_string += str(seq)
    if "n_epochs" in kwargs:
        exp_string += str(n_epochs)
    if "clahe_kwargs" in kwargs:
        clahe_kwargs = deepcopy(kwargs["clahe_kwargs"])
        if clahe_kwargs is None:
            exp_string += "clahe_kwargs"+str(None)
        else:
            clahe_kwargs["cl"] = clahe_kwargs.pop("clipLimit")
            clahe_kwargs["tgs"] = clahe_kwargs.pop("tileGridSize")
            exp_string += "clahe_kwargs"+json.dumps(clahe_kwargs, sort_keys=True)
    if "per_image_normalization" in kwargs:
        exp_string += "per_image_normalization"+str(kwargs["per_image_normalization"])
    if "gamma" in kwargs:
        exp_string += "gamma"+str(kwargs["gamma"])
    exp_string = exp_string.replace("\'","").replace("\"","").replace(",","").replace(" ","-").\
        replace("{","(").replace("}",")").replace("[","(").replace("]",")").replace(":","-")

    #exp_string = exp_string.replace("\\", "").replace('/', '')
    return exp_string

if __name__ == '__main__':

    ### DEBUG THIS TO CHECK EPOCH COUNTS
    ### WAS A BUG WHEN CALCULATING MEAN OF SCORE SHEETS
    ### AT MINIMUM SHOULD CHANGE SCORING FOR MORE ENTRIES (THOUGH THAT SHOULD BE FOR THE RESULST ANALYZER, NOT SURE ISSUE)

    EXPERIMENTS_DIR_PATH = "/root/vessel-segmentation/experiments11_test_eval"
    # EXPERIMENTS_DIR_PATH = "C:\\vessel-segmentation\\aug_exp1"

    metrics_epoch_freq = 5
    viz_layer_epoch_freq = 10001

    WRK_DIR_PATH = "/root/vessel-segmentation/DRIVE"
    #WRK_DIR_PATH = "C:\\vessel-segmentation\\drive"
    n_epochs = 200


    job_kwargs = {"tuning_constant":1.5, "objective_fn": "wce", "regularizer_args": ("L1",1E-8),
                  "op_fun_and_kwargs": ("rmsprop", {}),
                  "learning_rate_and_kwargs": (.001, {"decay_epochs": 50, "decay_rate": .1, "staircase": True}),
                  "weight_init": "default", "act_fn": "lrelu", "act_leak_prob": 0.2, "hist_eq": False,
                  "clahe_kwargs": {"clipLimit": 2.0,"tileGridSize":(8,8)}, "per_image_normalization": True,
                  "gamma": 2.0,
                  "seq": iaa.Sequential([iaa.Affine(rotate=(-90, 90), mode='constant', cval=0, name="rotate")])}

    kwarg_options = [{"tuning_constant": 1.75},  {"tuning_constant": 2.25}, {"tuning_constant": 2.5},
                     {"seq": iaa.Sequential([iaa.Affine(rotate=(-27.5, 27.5), mode='constant', cval=0, name="rotate27_5")])},
                     {"seq": iaa.Sequential([iaa.Affine(rotate=(-67.5, 67.5), mode='constant', cval=0, name="rotate67_ 5")])}
                     ]

    for kwarg_option in kwarg_options:
        current_job_kwargs = deepcopy(job_kwargs)
        for key, value in kwarg_option.items():
            current_job_kwargs[key] = value

        EXPERIMENT_NAME = get_experiment_string(**kwarg_option)
        OUTPUTS_DIR_PATH = os.path.join(EXPERIMENTS_DIR_PATH, EXPERIMENT_NAME)


        job = DriveJob(OUTPUTS_DIR_PATH=OUTPUTS_DIR_PATH)
        job.run_single_model(WRK_DIR_PATH=WRK_DIR_PATH, early_stopping=False, early_stopping_metric="auc",
                             save_model=False, save_sample_test_images=False, metrics_epoch_freq=metrics_epoch_freq,
                             viz_layer_epoch_freq=viz_layer_epoch_freq, n_epochs=n_epochs, **current_job_kwargs)