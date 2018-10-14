from job.dsa import DsaJob
from job.drive import DriveJob
from job.stare import StareJob
from job.chase import ChaseJob
from imgaug import augmenters as iaa
from itertools import product
from random import sample
import os
import json

def get_experiment_string(objective_fn,tuning_constant,ss_r,regularizer_args,op_fun_and_kwargs,
                          learning_rate_and_kwargs, weight_init, act_fn, act_leak_prob, seq, hist_eq, clahe_kwargs,
                          per_image_normalization,gamma):
    exp_string = ""
    exp_string += objective_fn + "|"
    exp_string += str(tuning_constant) + "|"
    exp_string += str(ss_r) if objective_fn=="ss" else str(None) + "|"
    exp_string += str(regularizer_args) + "|"
    exp_string += str(op_fun_and_kwargs) + "|"

    learning_rate_kwargs = learning_rate_and_kwargs[1]
    if "decay_epochs" in learning_rate_kwargs:
        learning_rate_kwargs["d_e"]=learning_rate_kwargs.pop("decay_epochs")
    if "decay_rate" in learning_rate_kwargs:
        learning_rate_kwargs["d_r"] = learning_rate_kwargs.pop("decay_rate")
    if "staircase" in learning_rate_kwargs:
        learning_rate_kwargs["s_c"] = learning_rate_kwargs.pop("staircase")
    exp_string += "("+str(learning_rate_and_kwargs[0]) +","+json.dumps(learning_rate_kwargs, sort_keys=True) + ")|"

    exp_string += weight_init + "|"
    exp_string += act_fn + "|"
    exp_string += str(act_leak_prob) + "|"
    exp_string += str(seq)
    exp_string += str(hist_eq) + "|"

    if clahe_kwargs is None:
        exp_string += str(None) + "|"
    else:
        clahe_kwargs["cl"] = clahe_kwargs.pop("clipLimit")
        clahe_kwargs["tgs"] = clahe_kwargs.pop("tileGridSize")
        exp_string += json.dumps(clahe_kwargs, sort_keys=True) + "|"

    exp_string += str(per_image_normalization) + "|"
    exp_string += str(gamma)

    return exp_string

if __name__ == '__main__':

    num_searches = 20
    EXPERIMENTS_DIR_PATH = "/home/ubuntu/new_vessel_segmentation/vessel-segmentation/experiments3"

    metrics_epoch_freq = 5
    viz_layer_epoch_freq = 101
    n_epochs = 100

    WRK_DIR_PATH = "/home/ubuntu/new_vessel_segmentation/vessel-segmentation/drive"
    n_splits = 4

    ### RANDOM SEARCH
    tuning_constants = [.5,1.0,1.5,2.0]
    ss_rs = [.166,.5,.6,.667]
    objective_fns = ["wce","gdice","ss"]
    regularizer_argss = [None, None, None,("L1",1E-6), ("L1",1E-8), ("L2",1E-4),("L2",1E-6),("L2",1E-6),("L2",1E-6), ("L2",1E-8)]
    learning_rate_and_kwargss = [(.1, {"decay_epochs":5,"decay_rate":.1,"staircase":False}),
                                 (.1, {"decay_epochs":5,"decay_rate":.1,"staircase":True}),
                                 (.1, {"decay_epochs": 10, "decay_rate": .1, "staircase": True}),
                                 (.1, {}),
                                 (.01, {}),
                                 (.001, {})]

    op_fun_and_kwargss = [("adam", {}), ("grad", {}), ("rmsprop", {}), ("rmsprop", {}), ("rmsprop", {})]
    weight_inits = ["default","He","Xnormal"]
    act_fns = ["lrelu"]
    act_leak_probs = [0.0,0.2,0.2,0.4,0.6]

    hist_eqs = [False]

    clahe_kwargss = [None, {"clipLimit": 2.0,"tileGridSize":(8,8)}, {"clipLimit": 2.0,"tileGridSize":(4,4)},
                     {"clipLimit": 2.0,"tileGridSize":(16,16)}, {"clipLimit": 20.0, "tileGridSize": (8, 8)},
                     {"clipLimit": 60.0, "tileGridSize": (8, 8)}]

    per_image_normalizations = [False, True]
    gammas = [1.0,2.0,6.0]

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

