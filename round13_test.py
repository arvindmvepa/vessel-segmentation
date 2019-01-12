from job.drive import DriveJob, DriveCustomJob
from itertools import product
from random import sample
import os
import json
from copy import deepcopy
import csv
from utilities.misc import get_job_kwargs_from_job_opts
from imgaug import augmenters as iaa
from imgaug import parameters as iap

def get_experiment_string(objective_fn, tuning_constant, ss_r, regularizer_args, op_fun_and_kwargs,
                          learning_rate_and_kwargs, weight_init, act_fn, act_leak_prob, seq, hist_eq, clahe_kwargs,
                          per_image_normalization, gamma, n_epochs, sep="__"):
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

    exp_string += str(n_epochs)
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

    if seq is not None:
        for aug in list(seq):
            exp_string += aug.name
        exp_string += sep
    else:
        exp_string += str(seq) + sep

    exp_string = exp_string.replace("\'","").replace("\"","").replace(",","").replace(" ","-").\
        replace("{","(").replace("}",")").replace("[","(").replace("]",")").replace(":","-")

    #exp_string = exp_string.replace("\\", "").replace('/', '')
    return exp_string

if __name__ == '__main__':

    num_searches = 40
    EXPERIMENTS_DIR_PATH = "/home/ubuntu/new_vessel_segmentation/vessel-segmentation/experiments6_1"
    # EXPERIMENTS_DIR_PATH = "C:\\vessel-segmentation\\experiments6_1"

    metrics_epoch_freq = 5
    viz_layer_epoch_freq = 1000
    n_epochss = [100, 200]

    WRK_DIR_PATH = "/home/ubuntu/new_vessel_segmentation/vessel-segmentation/drive"
    #WRK_DIR_PATH = "C:\\vessel-segmentation\\drive"
    n_splits = 4

    ### RANDOM SEARCH
    tuning_constants = [.5,1.0,1.5,2.0]

    ss_rs = [None] # No impact
    objective_fns = ["wce"] # removed ss

    regularizer_argss = [None,("L1",1E-8),("L2",1E-4),("L2",1E-6), ("L2",1E-8)] # relevant factor, not removing

    learning_rate_and_kwargss = [(.1, {"decay_epochs": 10, "decay_rate": .1, "staircase": True}),
                                 (.01, {"decay_epochs": 25, "decay_rate": .1, "staircase": True}),
                                 (.01, {"decay_epochs": 25, "decay_rate": .1, "staircase": False}),
                                 (.01, {"decay_epochs": 50, "decay_rate": .1, "staircase": True}),
                                 (.01, {"decay_epochs": 50, "decay_rate": .1, "staircase": False}),
                                 (.001, {"decay_epochs": 25, "decay_rate": .1, "staircase": True}),
                                 (.001, {"decay_epochs": 50, "decay_rate": .1, "staircase": True}),
                                 (.001, {"decay_epochs": 50, "decay_rate": .1, "staircase": False}),
                                 (.001, {})] # removed some earlier ones

    op_fun_and_kwargss = [("adam", {}), ("rmsprop", {})] # decreased freq of rmsprop
    weight_inits = ["default","He","Xnormal"]
    act_fns = ["lrelu"]
    act_leak_probs = [0.2] # remove the others due to lack of impact

    hist_eqs = [False]

    clahe_kwargss = [None] # b/c lack of impact, removing

    per_image_normalizations = [False, True]
    gammas = [1.0] # removing others due to lack of impact

    seqs = [iaa.Sequential([iaa.Affine(rotate=(-45, 45), mode='constant', cval=0, name="rotate45")]),
            iaa.Sequential([iaa.Affine(rotate=(-135, 135), mode='constant', cval=0, name="rotate135")]),
            iaa.Sequential([iaa.Affine(rotate=(-180, 180), mode='constant', cval=0, name="rotate180")]),
            ]
    """
    zero_centers = [True, False]
    per_image_z_score_norms = [True, False]
    per_image_zero_centers = [True, False]
    per_image_zero_center_scales = [True, False]
    zero_center_scales = [True, False]
    z_score_norms = [True, False]
    centers = [True, False]
    pooling_methods = ["AVG", "MAX"]
    unpooling_methods = ["bilinear", "nearest_neighbor", "bicubic"]
    last_layer_op =["AVG", "MAX", "CONV"]
    num_prev_last_conv_output_channels = [1,5,10,25,50,100]
    add_decoder_layers_map = "" # one option is to add a lot of decoder layers at the end
                                # add a conv layer per section
                                # randomly add a conv layer
    remove_decoder_layers_names = "" # remove a conv layer per section, other stuff discussed below, randomly remove a conv layer
    job_clss = [DriveJob, DriveCustomJob]
    - w/o skip connections
    encoder_model_keys = ["densenet121","incresv2","incv3", "resnet50", "resnet50v2", "resnext50", "xception"]
    - w/ skip connections
    encoder_model_keys = ["densenet121" "resnet50", "resnet50v2", "resnext50"]
    # pattern: beg. of pooling layer, beg. of conv. layer
    batch_norm = [True, False]
    act_fns = ["relu", "lrelu", "elu", "maxout"]
        act_leak_prob = [.2, .4, .6, .8]
    layer_params
        add_to_input  # all layers, no layers, at the beg. pooling layer, beg. of conv. layer # following pooling layer
        concat_to_input  # exchange the add patterns with concat
        dp_rate  # apply 0,.1,.2 uniformly after convolutional layers, maybe apply more after more dense layers (like .5_
        center (override)  # check for applying center for the last layer or not
        dilation (with stride)  # symmetric and asymmetric
            dilation impacts receptive field immediately, subsequent conv layers, only increments by same amount
            unless you increase the receptive field continuously (same as downsampling)
            increased stride has a more pronounced impact on receptive field subsequently
            - options:
                - remove pooling and use dilated convolutions (consider reducing number of output channels for memory)
                - remove pooling and use strided convolutions (consider reducing number of output channels for memory)
                - remove pooling entirely (reduce number of output channels for memory)
                - try adding dilated convolutions to increase receptive field in general
                - try merging input from different layers (not possible right now)
                - (doesn't have to be symmetric)
    """

    # may use a json system to save hyper-parameters
    # may be better than using experiment names

    total_hyper_parameter_combos = list(product(tuning_constants, ss_rs, objective_fns, regularizer_argss, learning_rate_and_kwargss,
                                                op_fun_and_kwargss, weight_inits, act_fns, act_leak_probs, hist_eqs, clahe_kwargss,
                                                per_image_normalizations, gammas, seqs, n_epochss))

    cur_hyper_parameter_combos = sample(total_hyper_parameter_combos, num_searches)

    for tuning_constant, ss_r, objective_fn, regularizer_args, learning_rate_and_kwargs, op_fun_and_kwargs, weight_init,\
        act_fn, act_leak_prob, hist_eq, clahe_kwargs, per_image_normalization, gamma, seq, n_epochs in \
            cur_hyper_parameter_combos:

        EXPERIMENT_NAME = get_experiment_string(objective_fn,tuning_constant,ss_r,regularizer_args,op_fun_and_kwargs,
                                                learning_rate_and_kwargs, weight_init, act_fn, act_leak_prob, seq,
                                                hist_eq, clahe_kwargs, per_image_normalization, gamma, n_epochs)

        OUTPUTS_DIR_PATH = os.path.join(EXPERIMENTS_DIR_PATH, EXPERIMENT_NAME)

        # gpu_device='/gpu:1'

        job = job_cls(OUTPUTS_DIR_PATH=OUTPUTS_DIR_PATH)
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
