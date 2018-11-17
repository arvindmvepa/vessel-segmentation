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

from utilities.misc import remove_duplicates, copy_jobs, analyze, get_job_kwargs_from_job_opts, get_job_opts

def clean_str(val):
    str_val = str(val)
    return str_val.replace("\'","").replace("\"","").replace(",","").replace(" ","-").replace("{","(").\
        replace("}",")").replace("[","(").replace("]",")").replace(":","-")


# hyper-parameter ordering corresponding to the file name ordering must be established here
def get_hyp_opts(all_hyps = ("objective_fns", "tuning_constants", "ss_rs", "regularizer_argss", "op_fun_and_kwargss",
                                  "learning_rate_and_kwargss","weight_inits","act_fns","act_leak_probs","seqs",
                                  "hist_eqs","clahe_kwargss","per_image_normalizations","gammas")):

    # hyper-parameter search space
    tuning_constants = [.5,1.0,1.5,2.0]
    ss_rs = [.166,.5,.6,.63,.667]
    objective_fns = ["wce","gdice","ss"]

    regularizer_argss = [None,("L1",1E-6), ("L1",1E-8), ("L2",1E-4),("L2",1E-6),("L2",1E-6),("L2",1E-6), ("L2",1E-8)]

    learning_rate_and_kwargss = [(.1, {"decay_epochs":5,"decay_rate":.1,"staircase":False}),
                                 (.1, {"decay_epochs":5,"decay_rate":.1,"staircase":True}),
                                 (.1, {"decay_epochs": 10, "decay_rate": .1, "staircase": False}),
                                 (.1, {}),
                                 (.01, {}),
                                 (.1, {"decay_epochs": 10, "decay_rate": .1, "staircase": True}),
                                 (.01, {"decay_epochs": 25, "decay_rate": .1, "staircase": True}),
                                 (.01, {"decay_epochs": 25, "decay_rate": .1, "staircase": False}),
                                 (.01, {"decay_epochs": 50, "decay_rate": .1, "staircase": True}),
                                 (.01, {"decay_epochs": 50, "decay_rate": .1, "staircase": False}),
                                 (.001, {"decay_epochs": 25, "decay_rate": .1, "staircase": True}),
                                 (.001, {"decay_epochs": 50, "decay_rate": .1, "staircase": True}),
                                 (.001, {"decay_epochs": 50, "decay_rate": .1, "staircase": False}),
                                 (.001, {}), (.00001, {})]

    op_fun_and_kwargss = [("adam", {}), ("grad", {}), ("rmsprop", {})]
    weight_inits = ["default","He","Xnormal"]
    act_fns = ["lrelu"]
    act_leak_probs = [0.0,0.2,0.4,0.6]

    hist_eqs = [False]

    clahe_kwargss = [None, {"clipLimit": 2.0,"tileGridSize":(8,8)}, {"clipLimit": 2.0,"tileGridSize":(4,4)},
                     {"clipLimit": 2.0,"tileGridSize":(16,16)}, {"clipLimit": 20.0, "tileGridSize": (8, 8)},
                     {"clipLimit": 60.0, "tileGridSize": (8, 8)}]

    per_image_normalizations = [False, True]
    gammas = [1.0,2.0,3.0,6.0]

    seqs = [None]

    # create dictionary with hyper-parameter name as keys and the corresponding list of hyper-parameter options as
    # values
    all_hyps_opts=dict()
    for hyp in all_hyps:
        hyp_opts = locals()[hyp][:]
        #incorporate `None` option if `ss` objective function not used
        if hyp == "ss_rs" or  hyp == "tuning_constants":
            hyp_opts = hyp_opts+[None]
        elif hyp == "learning_rate_and_kwargss":
            hyp_opts_ = []
            for hyp_opt in hyp_opts:
                hyp_opt = deepcopy(hyp_opt)
                hyp_opt1 = hyp_opt[1]
                if "decay_epochs" in hyp_opt1:
                    hyp_opt1["d_e"] = hyp_opt1.pop("decay_epochs")
                if "decay_rate" in hyp_opt1:
                    hyp_opt1["d_r"] = hyp_opt1.pop("decay_rate")
                if "staircase" in hyp_opt1:
                    hyp_opt1["s_c"] = hyp_opt1.pop("staircase")
                hyp_opt = "(" + str(hyp_opt[0]) + "," + json.dumps(hyp_opt1,sort_keys=True) + ")"
                hyp_opts_.append(hyp_opt)
            hyp_opts = hyp_opts_
        elif hyp == "clahe_kwargss":
            hyp_opts_ = []
            for hyp_opt in hyp_opts:
                hyp_opt = deepcopy(hyp_opt)
                if hyp_opt is None:
                    hyp_opt = str(None)
                else:
                    hyp_opt["cl"] = hyp_opt.pop("clipLimit")
                    hyp_opt["tgs"] = hyp_opt.pop("tileGridSize")
                    hyp_opt = json.dumps(hyp_opt, sort_keys=True)
                hyp_opts_ += [hyp_opt]
            hyp_opts = hyp_opts_
        hyp_opts = [clean_str(hyp_opt) for hyp_opt in hyp_opts]
        #remove any duplicates
        hyp_opts = remove_duplicates(hyp_opts)
        all_hyps_opts[hyp] = hyp_opts

    return all_hyps_opts, all_hyps

if __name__ == '__main__':
    EXPERIMENTS_DIR_PATH = "/Users/arvind.m.vepa/Documents/vessel segmentation/round7_test/overall results"
    OUTPUT_DIR_PATH = "/Users/arvind.m.vepa/Documents/vessel segmentation/round7_test/overall output"
    TOP_EXPERIMENTS_DIR_PATH = "/Users/arvind.m.vepa/Documents/vessel segmentation/round7_test/top overall results"
    TOP_OUTPUT_DIR_PATH = "/Users/arvind.m.vepa/Documents/vessel segmentation/round7_test/top overall output"

    jobs = analyze(EXPERIMENTS_DIR_PATH=EXPERIMENTS_DIR_PATH, OUTPUT_DIR_PATH=OUTPUT_DIR_PATH, get_hyp_opts=get_hyp_opts, rt_jobs_score=.97, file_char="log")
    copy_jobs(jobs, EXPERIMENTS_DIR_PATH, TOP_EXPERIMENTS_DIR_PATH)

    analyze(EXPERIMENTS_DIR_PATH=TOP_EXPERIMENTS_DIR_PATH, OUTPUT_DIR_PATH=TOP_OUTPUT_DIR_PATH, get_hyp_opts=get_hyp_opts, file_char="log")












