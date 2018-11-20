import numpy as np
from itertools import groupby
import shutil, errno
import os
import csv
from statsmodels import robust
from collections import defaultdict
import re
from copy import deepcopy
import json

def find_closest_pos(positions, start_pos=(0,0)):
    min = np.inf
    min_index = -1
    start_pos = np.array(start_pos)
    for i in range(len(positions)):
        pos = positions[i]
        test = np.linalg.norm(start_pos - pos)
        if test < min:
            min = test
            min_index = i
    min_pos = positions.pop(min_index)
    return min_pos, positions


def find_class_balance(targets, masks):
    total_pos = 0
    total_num_pixels = 0
    for target, mask in zip(targets, masks):
        target = np.multiply(target, mask)
        total_pos += np.count_nonzero(target)
        total_num_pixels += np.count_nonzero(mask)
    total_neg = total_num_pixels - total_pos
    weight = total_neg / total_pos
    return weight, float(total_neg)/float(total_num_pixels), float(total_pos)/float(total_num_pixels)

def remove_duplicates(data):
    ''' Remove duplicates from the data (normally a list).
        The data must be sortable and have an equality operator
    '''
    data = sorted(data)
    return [k for k, v in groupby(data)]

#https://stackoverflow.com/questions/1994488/copy-file-or-directories-recursively-in-python
def copy_stuff(src, dst):
    try:
        shutil.copytree(src, dst)
    except OSError as exc: # python >2.5
        if exc.errno == errno.ENOTDIR:
            shutil.copy(src, dst)
        else: raise

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

def get_job_kwargs_from_job_opts(job_opts):
    job_dict = dict()
    job_dict["objective_fn"] = job_opts[0]
    job_dict["tuning_constant"] = None if job_opts[1] == "None" else float(job_opts[1])
    job_dict["ss_r"] =  None if job_opts[2] == "None" else float(job_opts[2])

    if job_opts[3] == "None":
        job_dict["regularizer_args"] = None
    else:
        reg_str = job_opts[3][1:len(job_opts[3])-1]
        sep_index = reg_str.find("-")
        reg_list = [reg_str[:sep_index], reg_str[sep_index+1:]]
        reg_list[1] = float(reg_list[1])
        job_dict["regularizer_args"] = tuple(reg_list)

    opt_list = job_opts[4].replace("-", ",").replace(")","").replace("(","").split(",")
    opt_list[1]={}
    job_dict["op_fun_and_kwargs"] = tuple(opt_list)

    lr_str = job_opts[5][1:len(job_opts[5])-1]
    lr_list = lr_str.split("(")
    lr_list[0] = float(lr_list[0])
    lr_list[1] = lr_list[1].replace(")","")
    if lr_list[1] == "":
        lr_list[1] = {}
    else:
        decay_epochs = 'd_e--'
        decay_rate = '-d_r--'
        staircase = '-s_c--'
        de_index = lr_list[1].find(decay_epochs)
        dr_index = lr_list[1].find(decay_rate)
        sc_index = lr_list[1].find(staircase)
        lr_list[1] = {"decay_epochs": int(lr_list[1][de_index+len(decay_epochs):dr_index]),
                      "decay_rate": float(lr_list[1][dr_index+len(decay_rate):sc_index]),
                      "staircase": lr_list[1][sc_index+len(staircase):] == "true"}
    job_dict["learning_rate_and_kwargs"] = tuple(lr_list)

    job_dict["weight_init"] = job_opts[6]
    job_dict["act_fn"] = job_opts[7]
    job_dict["act_leak_prob"] = float(job_opts[8])
    job_dict["seq"] = None
    job_dict["hist_eq"] = job_opts[10] == 'True'

    if job_opts[11] == "None":
        job_dict["clahe_kwargs"] = None
    else:
        ck_str = job_opts[11][1:len(job_opts[11]) - 1]
        clipLimit = "cl--"
        tileGridSize = "-tgs--"
        cl_index = ck_str.find(clipLimit)
        tgs_index = ck_str.find(tileGridSize)
        cl = float(ck_str[cl_index+len(clipLimit):tgs_index])
        tgs = ck_str[tgs_index+len(tileGridSize):]
        tgs = tgs[1:len(tgs)-1].split("-")
        tgs = tuple([int(tgs_) for tgs_ in tgs])
        job_dict["clahe_kwargs"] = {"clipLimit": cl,
                                    "tileGridSize": tgs}

    job_dict["per_image_normalization"] = job_opts[12] == 'True'
    job_dict["gamma"] = float(job_opts[13])
    return job_dict

def get_job_opts(job_file):
    job_opts = job_file.split("__")
    if len(job_opts) == 13:
        if "(" in job_opts[2]:
            pos = job_opts[2].find("(")
        elif "N" in job_opts[2]:
            pos = job_opts[2].find("N")
        arg0 = job_opts[2][:pos]
        arg1 = job_opts[2][pos:]
        job_opts[2] = arg0
        job_opts.insert(3, arg1)
    return job_opts


def copy_jobs(jobs, EXPERIMENTS_DIR_PATH, TOP_EXPERIMENTS_DIR_PATH, top=None):
    if not os.path.exists(TOP_EXPERIMENTS_DIR_PATH):
        os.makedirs(TOP_EXPERIMENTS_DIR_PATH)
    if top is not None:
        filtered_jobs = jobs[:top]
    else:
        filtered_jobs = jobs
    for job in filtered_jobs:
        cur_loc = os.path.join(EXPERIMENTS_DIR_PATH, job)
        new_loc = os.path.join(TOP_EXPERIMENTS_DIR_PATH, job)
        copy_stuff(cur_loc, new_loc)

def analyze(relevant_hyps = ("objective_fns", "tuning_constants", "ss_rs", "regularizer_argss",
                             "op_fun_and_kwargss","learning_rate_and_kwargss","weight_inits","act_fns","act_leak_probs",
                             "seqs","hist_eqs","clahe_kwargss", "per_image_normalizations","gammas"), mof_metric="mad",
            n_metric_intervals=20, EXPERIMENTS_DIR_PATH=".", OUTPUT_DIR_PATH=".", rt_jobs_score = None, file_char = "mof",
            rt_jobs_metric_interval=None, get_hyp_opts = lambda *args, **kwargs: None):

    if not os.path.exists(OUTPUT_DIR_PATH):
        os.makedirs(OUTPUT_DIR_PATH)

    # define func for measure of fit
    if mof_metric == "mad":
        mof_func = robust.mad
    elif mof_metric == "std":
        mof_func = np.std

    # dict for marginal hyp job results
    auc_roc_marg_scores = defaultdict(lambda : defaultdict(lambda : defaultdict(list)))

    # dict for job results
    job_results = defaultdict(lambda : defaultdict(float))

    all_hyps_opts, all_hyps = get_hyp_opts(all_hyps=relevant_hyps)

    job_files = os.listdir(EXPERIMENTS_DIR_PATH)

    for job_file in job_files:
        JOB_PATH = os.path.join(EXPERIMENTS_DIR_PATH, job_file)
        # get the file with the combined metrics score
        job_metrics_file = [file for file in os.listdir(JOB_PATH) if file_char in file][0]
        JOB_METRICS_PATH = os.path.join(JOB_PATH, job_metrics_file)
        # parse the hyper-parameter options from the file name
        job_opts = get_job_opts(job_file)

        auc_col = 1
        p = re.compile(r'\d+\.\d+')

        with open(JOB_METRICS_PATH) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            next(csv_reader, None)
            for i, row in enumerate(csv_reader):
                auc_results = row[auc_col]
                auc_results = [float(result) for result in p.findall(auc_results)][0:2]
                job_results[i][job_file] = auc_results
                for hyp_name, job_opt in zip(all_hyps, job_opts):
                    #print('debug')
                    #print(hyp_name)
                    #print(job_opt)

                    check = False
                    hyp_opts = all_hyps_opts[hyp_name]
                    #print(hyp_opts)
                    for hyp_opt in hyp_opts:
                        if str(hyp_opt) in job_opt or job_opt in str(hyp_opt):
                            auc_roc_marg_scores[i][hyp_name][str(hyp_opt)] = \
                                auc_roc_marg_scores[i][hyp_name][str(hyp_opt)] + [auc_results[0]]
                            check=True
                if not check:
                    print("missed {}, job opt str {}, parameterized opts {}".format(hyp_name, job_opt,
                                                                                    all_hyps_opts[hyp_name]))
    n_results_hyp_opts = []
    for i in range(n_metric_intervals):
        i_results = []
        i_hyp_opts = []
        for hyp_name in relevant_hyps:
            hyp_opts = all_hyps_opts[hyp_name]
            for hyp_opt in hyp_opts:
                if str(hyp_opt) in auc_roc_marg_scores[i][hyp_name]:
                    list_results = auc_roc_marg_scores[i][hyp_name][str(hyp_opt)]
                    mean_result = np.mean(list_results)
                    mof_result = mof_func(list_results)
                    i_results += [(mean_result, mof_result)]
                    i_hyp_opts += [hyp_name+"_"+str(hyp_opt)+" ({})".format(len(list_results))]
        n_results_hyp_opts += [zip(i_results, i_hyp_opts)]


    hyp_metrics_log = "marg_hyp_log.csv"
    hyp_metrics_log_path = os.path.join(OUTPUT_DIR_PATH, hyp_metrics_log)

    with open(hyp_metrics_log_path, "a") as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        writer.writerow([i_result_hyp_opt[1] for i_result_hyp_opt in n_results_hyp_opts[0]]+["mean", mof_metric])
        for i in range(n_metric_intervals):
            i_results, _= zip(*n_results_hyp_opts[i])
            mean_results, _ = zip(*i_results)
            writer.writerow([" +/- ".join([str(metric) for metric in np.round(i_result,4)])+" % rank {:.1%}".format(float(i)/len(i_results)) for i,i_result in enumerate(i_results)]+
                            [np.round(np.mean(mean_results),4),np.round(mof_func(mean_results),4)])

    rank_hyp_metrics_log = "rank_marg_hyp_log.csv"
    rank_hyp_metrics_log_path = os.path.join(OUTPUT_DIR_PATH, rank_hyp_metrics_log)

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
    rank_job_log_path = os.path.join(OUTPUT_DIR_PATH, rank_job_log)

    rt_jobs = []
    with open(rank_job_log_path, "a") as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        for i in range(n_metric_intervals):
            i_job_results = job_results[i]
            i_job_results = i_job_results.items()
            i_job_results = sorted(i_job_results, key = lambda x: x[1][0], reverse=True)
            i_jobs, i_results = zip(*i_job_results)
            if file_char == "mof":
                mean_results, _ = zip(*i_results)
            else:
                mean_results = zip(*i_results)
            if rt_jobs_score is not None and (rt_jobs_metric_interval is None or i == rt_jobs_metric_interval):
                for i_job, i_result in zip(i_jobs, i_results):
                    if i_result[0] > rt_jobs_score:
                        if i_job not in rt_jobs:
                            rt_jobs.append(i_job)
            elif rt_jobs_metric_interval is not None and i == rt_jobs_metric_interval:
                rt_jobs = i_jobs
            writer.writerow(list(i_jobs)+["mean", mof_metric])
            writer.writerow(list([" +/- ".join([str(metric) for metric in np.round(i_result,4)])+" % rank {:.1%}".format(float(i)/len(i_results)) for i,i_result in enumerate(i_results)])+[np.round(np.mean(mean_results),4),np.round(mof_func(mean_results),4)])

    return rt_jobs