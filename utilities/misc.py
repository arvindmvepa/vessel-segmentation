import numpy as np
from itertools import groupby
import collections
import shutil, errno
import os
import csv
from statsmodels import robust
from collections import defaultdict
import re


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


def update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.Mapping):
            d[k] = update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


#https://stackoverflow.com/questions/1994488/copy-file-or-directories-recursively-in-python
def copy_stuff(src, dst):
    try:
        shutil.copytree(src, dst)
    except OSError as exc: # python >2.5
        if exc.errno == errno.ENOTDIR:
            shutil.copy(src, dst)
        else: raise

def copy_jobs(jobs, EXPERIMENTS_DIR_PATH, TOP_EXPERIMENTS_DIR_PATH, top=12):
    if not os.path.exists(TOP_EXPERIMENTS_DIR_PATH):
        os.makedirs(TOP_EXPERIMENTS_DIR_PATH)
    filtered_jobs = jobs[:top]
    for job in filtered_jobs:
        cur_loc = os.path.join(EXPERIMENTS_DIR_PATH, job)
        new_loc = os.path.join(TOP_EXPERIMENTS_DIR_PATH, job)
        copy_stuff(cur_loc, new_loc)

def analyze(relevant_hyps = ("objective_fns", "tuning_constants", "ss_rs", "regularizer_argss",
                             "op_fun_and_kwargss","learning_rate_and_kwargss","weight_inits","act_fns","act_leak_probs",
                             "seqs","hist_eqs","clahe_kwargss", "per_image_normalizations","gammas"), mof_metric="mad",
            n_metric_intervals=20, EXPERIMENTS_DIR_PATH=".", OUTPUT_DIR_PATH=".", rt_jobs_metric_interval=10,
            get_hyp_opts = lambda *args, **kwargs: None):

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
        job_metrics_file = [file for file in os.listdir(JOB_PATH) if "mof" in file][0]
        JOB_METRICS_PATH = os.path.join(JOB_PATH, job_metrics_file)
        # parse the hyper-parameter options from the file name
        job_opts = job_file.split("__")
        if len(job_opts) == 13:
            if "(" in job_opts[2]:
                pos = job_opts[2].find("(")
            elif "N" in job_opts[2]:
                pos = job_opts[2].find("N")
            arg0 = job_opts[2][:pos]
            arg1 = job_opts[2][pos:]
            job_opts[2] = arg0
            job_opts.insert(3,arg1)

        auc_col = 1
        p = re.compile(r'\d+\.\d+')

        with open(JOB_METRICS_PATH) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            next(csv_reader, None)
            for i, row in enumerate(csv_reader):
                auc_results = row[auc_col]
                auc_results = [float(result) for result in p.findall(auc_results)][0:2]
                job_results[i][job_file] = auc_results
                for hyp_name,job_opt in zip(all_hyps, job_opts):
                    check = False
                    hyp_opts = all_hyps_opts[hyp_name]
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

    rt_jobs = None
    with open(rank_job_log_path, "a") as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        for i in range(n_metric_intervals):
            i_job_results = job_results[i]
            i_job_results = i_job_results.items()
            i_job_results = sorted(i_job_results, key = lambda x: x[1][0], reverse=True)
            i_job, i_results = zip(*i_job_results)
            mean_results, _ = zip(*i_results)
            if i == rt_jobs_metric_interval:
                rt_jobs = i_job
            writer.writerow(list(i_job)+["mean", mof_metric])
            writer.writerow(list([" +/- ".join([str(metric) for metric in np.round(i_result,4)])+" % rank {:.1%}".format(float(i)/len(i_results)) for i,i_result in enumerate(i_results)])+[np.round(np.mean(mean_results),4),np.round(mof_func(mean_results),4)])

    return rt_jobs
