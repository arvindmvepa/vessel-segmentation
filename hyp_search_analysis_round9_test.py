#from imgaug import augmenters as iaa
import csv
from collections import defaultdict
import numpy as np
import re
from utilities.misc import remove_duplicates
from statsmodels import robust
from copy import deepcopy
import json
import os
from imgaug import augmenters as iaa
from imgaug import parameters as iap

from utilities.misc import remove_duplicates, copy_jobs, analyze, get_job_kwargs_from_job_opts, get_job_opts

def clean_str(val):
    str_val = str(val)
    return str_val.replace("\'","").replace("\"","").replace(",","").replace(" ","-").replace("{","(").\
        replace("}",")").replace("[","(").replace("]",")").replace(":","-")


# hyper-parameter ordering corresponding to the file name ordering must be established here
def get_hyp_opts(all_hyps = ("seqs", "num_epochss")):

    # hyper-parameter search space

    num_epochss = [100,200]
    seqs = [iaa.Sequential([iaa.Fliplr(0.5, name="Flipper")]),
            iaa.Sequential([iaa.GaussianBlur(sigma=(0, 3.0), name="GaussianBlur")]),
            iaa.Sequential([iaa.Affine(rotate=iap.Choice([0,90,180,270]), mode='constant', cval=0, name="rotateNoInterp")]),
            iaa.Sequential([iaa.Affine(rotate=(-90, 90), mode='constant', cval=0, name="rotate")]),
            iaa.Sequential([iaa.Affine(rotate=iap.Choice([0,90,180,270]), mode='constant', cval=0, name="rotateNoInterp"),
                            iaa.Fliplr(0.5, name="Flipper")]),
            iaa.Sequential([iaa.Affine(rotate=(-90, 90), mode='constant', cval=0, name="rotate"),
                            iaa.Fliplr(0.5, name="Flipper")]),
            iaa.Sequential([iaa.Affine(rotate=(-10, 10), mode='constant', cval=0, name="rotate_small")]),
            None
            ]

    # create dictionary with hyper-parameter name as keys and the corresponding list of hyper-parameter options as
    # values
    all_hyps_opts=dict()
    for hyp in all_hyps:
        hyp_opts = locals()[hyp][:]
        if hyp == "seqs":
            hyp_opts_ = []
            for hyp_opt in hyp_opts:
                hyp_opt_ = ""
                if hyp_opt is not None:
                    for aug in list(hyp_opt):
                        hyp_opt_ += aug.name
                else:
                    hyp_opt_ += str(hyp_opt)
                hyp_opts_ += [hyp_opt_]
            hyp_opts = hyp_opts_
        if hyp == "n_epochss":
            hyp_opts = [str(hyp_opt) for hyp_opt in hyp_opts]

        hyp_opts = [clean_str(hyp_opt) for hyp_opt in hyp_opts]
        #remove any duplicates
        hyp_opts = remove_duplicates(hyp_opts)
        all_hyps_opts[hyp] = hyp_opts

    return all_hyps_opts, all_hyps

if __name__ == '__main__':
    model = 2
    num_epochs = 200
    MAIN_DIR = "/Users/arvind.m.vepa/Documents/vessel segmentation/"
    DIR_EXT = "round"
    round = 9
    n_metric_intervals = num_epochs/5

    overall_results = "overall results"
    overall_output = "overall output"

    top_overall_results = "top overall results"
    top_overall_output = "top overall output"

    EXPERIMENTS_DIR_PATH = MAIN_DIR + DIR_EXT+ str(round) + "_"+str(model) + "_test/"+str(num_epochs)+"/"+str(overall_results)
    OUTPUT_DIR_PATH = MAIN_DIR + DIR_EXT+ str(round) + "_"+str(model) + "_test/"+str(num_epochs)+"/"+str(overall_output)
    TOP_EXPERIMENTS_DIR_PATH = MAIN_DIR + DIR_EXT+ str(round) + "_"+str(model) + "_test/"+str(num_epochs)+"/"+str(top_overall_results)
    TOP_OUTPUT_DIR_PATH = MAIN_DIR + DIR_EXT+ str(round) + "_"+str(model) + "_test/"+str(num_epochs)+"/"+str(top_overall_output)

    jobs = analyze(EXPERIMENTS_DIR_PATH=EXPERIMENTS_DIR_PATH, OUTPUT_DIR_PATH=OUTPUT_DIR_PATH, get_hyp_opts=get_hyp_opts, rt_jobs_score=.97, relevant_hyps=("seqs","num_epochss"), file_char="log", n_metric_intervals=n_metric_intervals)
    copy_jobs(jobs, EXPERIMENTS_DIR_PATH, TOP_EXPERIMENTS_DIR_PATH)
    """
    # save jobs that score above .970 AUC
    with open("top_job_opts.csv", "a") as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        for job in jobs:
            job_opts = get_job_opts(job)
            print(get_job_kwargs_from_job_opts(job_opts))
            writer.writerow(job_opts)
    """
    analyze(EXPERIMENTS_DIR_PATH=TOP_EXPERIMENTS_DIR_PATH, OUTPUT_DIR_PATH=TOP_OUTPUT_DIR_PATH, get_hyp_opts=get_hyp_opts, relevant_hyps=("seqs", "num_epochss"), file_char="log", n_metric_intervals=n_metric_intervals)












