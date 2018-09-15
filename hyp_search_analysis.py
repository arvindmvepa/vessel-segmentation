#from imgaug import augmenters as iaa
import csv
from collections import defaultdict
import numpy as np
import re
from utilities.misc import remove_duplicates

import os

# hyper-parameter ordering corresponding to the file name ordering must be established here
def get_hyp_opts(all_hyps = ("objective_fns", "tuning_constants", "ss_rs", "regularizer_argss", "op_fun_and_kwargss",
                                  "learning_rate_and_kwargss","weight_inits","act_fns","act_leak_probs","seqs",
                                  "hist_eqs","clahe_kwargss","per_image_normalizations","gammas")):

    # hyper-parameter search space
    tuning_constants = [.2,.5,1.0,1.5,2.0]
    ss_rs = [.166,.33,.5,.6,.667]
    objective_fns = ["wce","gdice","ss"]
    regularizer_argss = [None, None,None, None,("L1",1E-8),("L1",1E-6),("L1",1E-4),("L1",1E-2),("L2",1E-8),("L2",1E-6),
                        ("L2",1E-4),("L2",1E-2)]
    learning_rate_and_kwargss = [(.1, {"decay_epochs":5,"decay_rate":.1,"staircase":False}),
                                 (.1, {"decay_epochs":5,"decay_rate":.1,"staircase":True}),
                                 (.1, {"decay_epochs": 10, "decay_rate": .1, "staircase": False}),
                                 (.1, {"decay_epochs": 10, "decay_rate": .1, "staircase": True}),
                                 (.1, {}),
                                 (.01, {}),
                                 (.001, {})]

    op_fun_and_kwargss = [("adam", {}), ("grad", {}), ("adagrad", {}), ("adadelta", {}), ("rmsprop", {})]
    weight_inits = ["default","He","Xnormal"]
    act_fns = ["lrelu"]
    act_leak_probs = [0.0,0.0,0.2,.2,0.4,0.6]
    hist_eqs = [True,False]
    clahe_kwargss = [None, None, None, None, None,
                     {"clipLimit": 2.0,"tileGridSize":(8,8)}, {"clipLimit": 2.0,"tileGridSize":(4,4)},
                     {"clipLimit": 2.0,"tileGridSize":(16,16)}, {"clipLimit": 20.0, "tileGridSize": (8, 8)},
                     {"clipLimit": 60.0, "tileGridSize": (8, 8)}]
    per_image_normalizations = [False, True]
    gammas = [1.0,1.0,1.0,2.0,4.0,6.0]
    seqs = [None, None, True]

    # create dictionary with hyper-parameter name as keys and the corresponding list of hyper-parameter options as
    # values
    hyps_opts=dict()
    for hyp in all_hyps:
        # update name for complex image augmentation name
        if hyp == "seqs":
            hyp_opts=[False, True]
        #incorporate `None` option if `ss` objective function not used
        elif hyp == "ss_rs":
            hyp_opts = locals()[hyp][:]+[None]
        else:
            hyp_opts = locals()[hyp][:]
        #remove any duplicates
        hyp_opts = remove_duplicates(hyp_opts)
        #remove any unnecessary string characters
        hyps_opts[hyp]= [str(hyp_opt).replace('"', '').replace("'", '') for hyp_opt in hyp_opts]

    return hyps_opts, all_hyps

def analyze(relevant_hyps = ("objective_fns", "tuning_constants", "ss_rs", "regularizer_argss",
                             "op_fun_and_kwargss","learning_rate_and_kwargss","weight_inits","act_leak_probs","seqs",
                             "hist_eqs","clahe_kwargss","gammas")):
    # dict for Job results
    auc_roc_marg_scores = defaultdict(lambda : defaultdict(lambda : defaultdict(list)))
    n_metric_intervals = 4

    hyps_opts, all_hyps = get_hyp_opts()

    EXPERIMENTS_DIR_PATH = "/Users/arvind.m.vepa/Documents/vessel segmentation/first round hyp results/experiments1"
    job_files = os.listdir(EXPERIMENTS_DIR_PATH)

    for job_file in job_files:
        JOB_PATH = os.path.join(EXPERIMENTS_DIR_PATH, job_file)
        job_metrics_file = [file for file in os.listdir(JOB_PATH) if "mof" in file][0]
        JOB_METRICS_PATH = os.path.join(JOB_PATH, job_metrics_file)
        job_file_str = job_file[1:len(job_file)-1].replace('"', '').replace("'", '')
        job_opts = re.split(r',\s*(?![^()]*\)|[^{}]*\})', job_file_str)

        auc_col = 1

        with open(JOB_METRICS_PATH) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            next(csv_reader, None)
            for i, row in enumerate(csv_reader):
                auc = row[auc_col]
                for hyp_name,job_opt in zip(all_hyps,job_opts):
                    check = False
                    if hyp_name in relevant_hyps:
                        hyp_opts = hyps_opts[hyp_name]
                        for hyp_opt in hyp_opts:
                            if str(hyp_opt) in job_opt or job_opt in str(hyp_opt):
                                auc_roc_marg_scores[i][hyp_name][str(hyp_opt)] = \
                                    auc_roc_marg_scores[i][hyp_name][str(hyp_opt)] + [auc]
                                check=True
                    if not check:
                        print("missed {}, job opt str {}, parameterized opts {}".format(hyp_name, job_opt,
                                                                                        hyps_opts[hyp_name]))

    hyp_metrics_log = "hyp_log.csv"
    hyp_metrics_log_path = os.path.join("/Users/arvind.m.vepa/Documents/vessel segmentation/first round hyp results",
                                        hyp_metrics_log)

    with open(hyp_metrics_log_path, "a") as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        hyp_keys_opts_strs = sum([[hyp_key+"_"+str(hyp_opt) for hyp_opt in auc_roc_marg_scores[0][hyp_key].keys()]
                              for hyp_key in relevant_hyps],[])
        writer.writerow(hyp_keys_opts_strs)

        p = re.compile(r'\d+\.\d+')
        for i in range(n_metric_intervals):
            metric_i_auc_roc_marg_scores = auc_roc_marg_scores[i]
            results = []
            for hyp_name in relevant_hyps:
                hyp_opts = hyps_opts[hyp_name]
                for hyp_opt in hyp_opts:
                    if str(hyp_opt) in auc_roc_marg_scores[i][hyp_name]:
                        list_results_str = auc_roc_marg_scores[i][hyp_name][str(hyp_opt)]
                        list_results = [float(p.findall(results_str)[0]) for results_str in list_results_str]
                        results += [np.mean(list_results)]
            writer.writerow(results)

if __name__ == '__main__':
    analyze()














