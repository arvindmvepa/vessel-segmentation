#from imgaug import augmenters as iaa
import csv
from collections import defaultdict
import numpy as np
import re
from utilities.misc import remove_duplicates
from statsmodels import robust

import os
import json

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
    """
    learning_rate_and_kwargss = [(.1, {"decay_epochs":5,"decay_rate":.1,"staircase":False}),
                                 (.1, {"decay_epochs":5,"decay_rate":.1,"staircase":True}),
                                 (.1, {"decay_epochs": 10, "decay_rate": .1, "staircase": False}),
                                 (.1, {"decay_epochs": 10, "decay_rate": .1, "staircase": True}),
                                 (.1, {}),
                                 (.01, {}),
                                 (.001, {})]
    """
    learning_rate_and_kwargss = [(.1, {"decayepochs":5,"decayrate":.1,"staircase":False}),
                                 (.1, {"decayepochs":5,"decayrate":.1,"staircase":True}),
                                 (.1, {"decayepochs": 10, "decayrate": .1, "staircase": False}),
                                 (.1, {"decayepochs": 10, "decayrate": .1, "staircase": True}),
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
                             "hist_eqs","clahe_kwargss","gammas"), mof_metric="mad", n_metric_intervals=4,
            EXPERIMENTS_DIR_PATH="/Users/arvind.m.vepa/Documents/vessel segmentation/second round hyp results/filtered lab comp"):

    # define func for measure of fit
    if mof_metric == "mad":
        mof_func = robust.mad
    elif mof_metric == "std":
        mof_func = np.std

    # dict for marginal hyp job results
    auc_roc_marg_scores = defaultdict(lambda : defaultdict(lambda : defaultdict(list)))
    # dict for job results
    job_results = defaultdict(lambda : defaultdict(float))

    hyps_opts, all_hyps = get_hyp_opts()

    job_files = os.listdir(EXPERIMENTS_DIR_PATH)

    for job_file in job_files:
        JOB_PATH = os.path.join(EXPERIMENTS_DIR_PATH, job_file)
        job_metrics_file = [file for file in os.listdir(JOB_PATH) if "mof" in file][0]
        JOB_METRICS_PATH = os.path.join(JOB_PATH, job_metrics_file)
        job_file_str = job_file[1:len(job_file)-1].replace("_","'").replace("$",":").replace('"', '').replace("'", '')
        #job_file_str = job_file[1:len(job_file) - 1].replace("$", "/").replace('"', '').replace("'",'')
        job_opts = re.split(r',\s*(?![^()]*\)|[^{}]*\})', job_file_str)
        auc_col = 1
        p = re.compile(r'\d+\.\d+')

        with open(JOB_METRICS_PATH) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            next(csv_reader, None)
            for i, row in enumerate(csv_reader):
                auc_results = row[auc_col]
                auc_results = [float(result) for result in p.findall(auc_results)][0:2]
                job_results[i][job_file_str] = auc_results
                for hyp_name,job_opt in zip(all_hyps,job_opts):
                    check = False
                    if hyp_name in relevant_hyps:
                        hyp_opts = hyps_opts[hyp_name]
                        for hyp_opt in hyp_opts:
                            if isinstance(hyp_opt,dict):
                                hyp_opt = json.dumps(hyp_opt, sort_keys=True)
                            else:
                                hyp_opt = str(hyp_opt)
                            if hyp_opt in job_opt or job_opt in hyp_opt:
                                auc_roc_marg_scores[i][hyp_name][hyp_opt] = \
                                    auc_roc_marg_scores[i][hyp_name][hyp_opt] + [auc_results[0]]
                                check=True
                    if not check:
                        print("missed {}, job opt str {}, parameterized opts {}".format(hyp_name, job_opt,
                                                                                        hyps_opts[hyp_name]))
    n_results_hyp_opts = []
    for i in range(n_metric_intervals):
        i_results = []
        i_hyp_opts = []
        for hyp_name in relevant_hyps:
            hyp_opts = hyps_opts[hyp_name]
            for hyp_opt in hyp_opts:
                if str(hyp_opt) in auc_roc_marg_scores[i][hyp_name]:
                    list_results = auc_roc_marg_scores[i][hyp_name][str(hyp_opt)]
                    mean_result = np.mean(list_results)
                    mof_result = mof_func(list_results)
                    i_results += [(mean_result, mof_result)]
                    i_hyp_opts += [hyp_name+"_"+str(hyp_opt)]
        n_results_hyp_opts += [zip(i_results,i_hyp_opts)]


    hyp_metrics_log = "marg_hyp_log.csv"
    hyp_metrics_log_path = os.path.join(EXPERIMENTS_DIR_PATH,hyp_metrics_log)

    with open(hyp_metrics_log_path, "a") as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        writer.writerow([i_result_hyp_opt[1] for i_result_hyp_opt in n_results_hyp_opts[0]]+["mean", mof_metric])
        for i in range(n_metric_intervals):
            i_results, _= zip(*n_results_hyp_opts[i])
            mean_results, _ = zip(*i_results)
            writer.writerow([" +/- ".join([str(metric) for metric in np.round(i_result,4)])+" % rank {:.1%}".format(float(i)/len(i_results)) for i,i_result in enumerate(i_results)]+
                            [np.round(np.mean(mean_results),4),np.round(mof_func(mean_results),4)])

    rank_hyp_metrics_log = "rank_marg_hyp_log.csv"
    rank_hyp_metrics_log_path = os.path.join(EXPERIMENTS_DIR_PATH, rank_hyp_metrics_log)

    with open(rank_hyp_metrics_log_path, "a") as csv_file:
        writer = csv.writer(csv_file, delimiter=',')

        for i in range(n_metric_intervals):
            i_results_hyp_opts = n_results_hyp_opts[i]
            i_results_hyp_opts = sorted(i_results_hyp_opts, key = lambda x: x[0][0], reverse=True)
            i_results, i_hyp_opts = zip(*i_results_hyp_opts)
            mean_results, _ = zip(*i_results)
            writer.writerow(list(i_hyp_opts)+["mean", mof_metric])
            writer.writerow([" +/- ".join([str(np.round(metric,4)) for metric in i_result])+" % rank {:.1%}".format(float(i)/len(i_results)) for i,i_result in enumerate(i_results)]+
                            [np.round(np.mean(mean_results),4),np.round(mof_func(mean_results),4)])

    rank_job_log = "rank_job_log.csv"
    rank_job_log_path = os.path.join(EXPERIMENTS_DIR_PATH, rank_job_log)

    with open(rank_job_log_path, "a") as csv_file:
        writer = csv.writer(csv_file, delimiter=',')

        for i in range(n_metric_intervals):
            i_job_results = job_results[i]
            i_job_results = i_job_results.items()
            i_job_results = sorted(i_job_results, key = lambda x: x[1][0], reverse=True)
            i_job, i_results = zip(*i_job_results)
            mean_results, _ = zip(*i_results)
            writer.writerow(list(i_job)+["mean", mof_metric])
            writer.writerow(list([" +/- ".join([str(metric) for metric in np.round(i_result,4)])+" % rank {:.1%}".format(float(i)/len(i_results)) for i,i_result in enumerate(i_results)])+[np.round(np.mean(mean_results),4),np.round(mof_func(mean_results),4)])

if __name__ == '__main__':
    analyze(n_metric_intervals=1)














