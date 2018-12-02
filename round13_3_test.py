from job.drive import DriveJob
from itertools import product
from random import sample
import os
import json
from copy import deepcopy
import csv
from utilities.misc import get_job_kwargs_from_job_opts
import math

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

    EXPERIMENTS_DIR_PATH = "/root/vessel-segmentation2/experiments13_3_test_eval"
    # EXPERIMENTS_DIR_PATH = "C:\\vessel-segmentation\\experiments6_1"

    metrics_epoch_freq = 5
    viz_layer_epoch_freq = 1000
    n_epochs = 100

    WRK_DIR_PATH = "/root/DRIVE"
    #WRK_DIR_PATH = "C:\\vessel-segmentation\\drive"

    job_opts_all = None
    with open('top_job_opts.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        job_opts_all = [row for row in csv_reader]

    num_jobs = len(job_opts_all)
    first_index = int(math.ceil(num_jobs/6.0))*2
    last_index = int(math.ceil(num_jobs/6.0))*3

    for job_opts in job_opts_all[first_index:last_index]:
        job_kwargs = get_job_kwargs_from_job_opts(job_opts)

        EXPERIMENT_NAME = get_experiment_string(**job_kwargs)

        OUTPUTS_DIR_PATH = os.path.join(EXPERIMENTS_DIR_PATH, EXPERIMENT_NAME)

        # gpu_device='/gpu:1'

        job = DriveJob(OUTPUTS_DIR_PATH=OUTPUTS_DIR_PATH)
        job.run_single_model(WRK_DIR_PATH=WRK_DIR_PATH, early_stopping=False, early_stopping_metric="auc",
                   save_model=False, save_sample_test_images=False, metrics_epoch_freq=metrics_epoch_freq,
                   viz_layer_epoch_freq=viz_layer_epoch_freq, n_epochs=n_epochs, **job_kwargs)

