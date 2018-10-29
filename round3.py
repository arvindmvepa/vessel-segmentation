from job.drive import DriveJob
from itertools import product
from random import sample
import os
import json
from copy import deepcopy

def get_experiment_string(objective_fn,tuning_constant,ss_r,regularizer_args,op_fun_and_kwargs,
                          learning_rate_and_kwargs, weight_init, act_fn, act_leak_prob, seq, hist_eq, clahe_kwargs,
                          per_image_normalization,gamma, sep="__"):
    exp_string = ""
    exp_string += objective_fn + sep
    exp_string += (str(tuning_constant) if objective_fn !="ss" else str(None)) + sep
    exp_string += (str(ss_r) if objective_fn=="ss" else str(None)) + sep
    exp_string += str(regularizer_args) + sep
    exp_string += str(op_fun_and_kwargs) + sep

    learning_rate_and_kwargs = deepcopy(learning_rate_and_kwargs)
    learning_rate_kwargs = learning_rate_and_kwargs[1]
    if "decay_epochs" in learning_rate_kwargs:
        learning_rate_kwargs["d_e"]=learning_rate_kwargs.pop("decay_epochs")
    if "decay_rate" in learning_rate_kwargs:
        learning_rate_kwargs["d_r"] = learning_rate_kwargs.pop("decay_rate")
    if "staircase" in learning_rate_kwargs:
        learning_rate_kwargs["s_c"] = learning_rate_kwargs.pop("staircase")
    exp_string += "("+str(learning_rate_and_kwargs[0]) +","+json.dumps(learning_rate_kwargs, sort_keys=True) + ")"+sep

    exp_string += weight_init + sep
    exp_string += act_fn + sep
    exp_string += str(act_leak_prob) + sep
    exp_string += str(seq) + sep
    exp_string += str(hist_eq) + sep

    clahe_kwargs = deepcopy(clahe_kwargs)
    if clahe_kwargs is None:
        exp_string += str(None) + sep
    else:
        clahe_kwargs["cl"] = clahe_kwargs.pop("clipLimit")
        clahe_kwargs["tgs"] = clahe_kwargs.pop("tileGridSize")
        exp_string += json.dumps(clahe_kwargs, sort_keys=True) + sep

    exp_string += str(per_image_normalization) + sep
    exp_string += str(gamma)
    exp_string = exp_string.replace("\'","").replace("\"","").replace(",","").replace(" ","-").\
        replace("{","(").replace("}",")").replace("[","(").replace("]",")").replace(":","-")

    #exp_string = exp_string.replace("\\", "").replace('/', '')
    return exp_string

if __name__ == '__main__':

    num_searches = 40
    EXPERIMENTS_DIR_PATH = "/home/ubuntu/new_vessel_segmentation/vessel-segmentation/experiments3"
    #EXPERIMENTS_DIR_PATH = "C:\\vessel-segmentation\\experiments3"

    metrics_epoch_freq = 5
    viz_layer_epoch_freq = 101
    n_epochs = 100

    WRK_DIR_PATH = "/home/ubuntu/new_vessel_segmentation/vessel-segmentation/drive"
    #WRK_DIR_PATH = "C:\\vessel-segmentation\\drive"
    n_splits = 4

    ### RANDOM SEARCH
    tuning_constants = [.5,1.0,1.5,1.5,1.5,2.0] # (increased 1.5 freq)
    ss_rs = [.6,.667, .8] # adjusted (removed lower constant values)
    objective_fns = ["wce","ss"] # adjusted (removed `gdice`)
    regularizer_argss = [None,("L1",1E-8),("L2",1E-4),("L2",1E-6), ("L2",1E-8)] # adjusted (removed some)
    learning_rate_and_kwargss = [(.01, {"decay_epochs": 5, "decay_rate": .1, "staircase": True}),
                                 (.01, {"decay_epochs": 5, "decay_rate": .1, "staircase": False}),
                                 (.01, {"decay_epochs": 10, "decay_rate": .1, "staircase": True}),
                                 (.01, {"decay_epochs": 10, "decay_rate": .1, "staircase": False}),
                                 (.001, {"decay_epochs": 5, "decay_rate": .1, "staircase": True}),
                                 (.001, {"decay_epochs": 5, "decay_rate": .1, "staircase": False}),
                                 (.001, {"decay_epochs": 10, "decay_rate": .1, "staircase": True}),
                                 (.001, {"decay_epochs": 10, "decay_rate": .1, "staircase": False}),
                                 (.001, {}),
                                 (.001, {}),
                                 (.001, {}),
                                 (.001, {}),
                                 (.001, {}),
                                 (.001, {}),
                                 (.00001, {})] # adjusted (increased .001 freq, added different lr schedules (and removed some))

    op_fun_and_kwargss = [("adam", {}), ("rmsprop", {})] # adjusted (removed grad)
    weight_inits = ["default","He","Xnormal"]
    act_fns = ["lrelu"]
    act_leak_probs = [0.0,0.2,0.2,0.4,0.6]

    hist_eqs = [False]

    clahe_kwargss = [None, None, None, {"clipLimit": 2.0,"tileGridSize":(8,8)}, {"clipLimit": 2.0,"tileGridSize":(4,4)},
                     {"clipLimit": 2.0,"tileGridSize":(16,16)}, {"clipLimit": 20.0, "tileGridSize": (8, 8)},
                     {"clipLimit": 60.0, "tileGridSize": (8, 8)}] # adjusted (decreased None freq)

    per_image_normalizations = [False, True]
    gammas = [1.0,1.0,2.0,6.0]

    seqs = [None]

    total_hyper_parameter_combos = list(product(tuning_constants, ss_rs, objective_fns, regularizer_argss, learning_rate_and_kwargss,
                                                op_fun_and_kwargss, weight_inits, act_fns, act_leak_probs, hist_eqs, clahe_kwargss,
                                                per_image_normalizations, gammas, seqs))

    cur_hyper_parameter_combos = sample(total_hyper_parameter_combos, num_searches)

    for tuning_constant, ss_r, objective_fn, regularizer_args, learning_rate_and_kwargs, op_fun_and_kwargs, weight_init,\
        act_fn, act_leak_prob, hist_eq, clahe_kwargs, per_image_normalization, gamma, seq in \
            cur_hyper_parameter_combos:

        EXPERIMENT_NAME = get_experiment_string(objective_fn,tuning_constant,ss_r,regularizer_args,op_fun_and_kwargs,
                                                learning_rate_and_kwargs, weight_init, act_fn, act_leak_prob, seq,
                                                hist_eq, clahe_kwargs, per_image_normalization, gamma)

        OUTPUTS_DIR_PATH = os.path.join(EXPERIMENTS_DIR_PATH, EXPERIMENT_NAME)

        job = DriveJob(OUTPUTS_DIR_PATH=OUTPUTS_DIR_PATH)
        job.run_cv(WRK_DIR_PATH=WRK_DIR_PATH, mc=True, val_prop=.10, early_stopping=False, early_stopping_metric="auc",
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

