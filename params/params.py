import yaml
import io
from utilities.misc import flatten, product_dict, update
from shutil import copyfile
from random import sample
import os
from glob import glob
from collections import defaultdict
import json
from statsmodels import robust
import numpy as np
import re
import csv

from job.drive import DriveJob, DriveCustomJob
from job.stare import StareJob
from job.chase import ChaseJob
from job.dsa import DsaJob

job_cls_map = {"DriveJob": DriveJob,
               "DriveCustomJob": DriveCustomJob,
               "ChaseJob": ChaseJob,
               "StareJob": StareJob,
               "DsaJob": DsaJob,
               }

metric_col_map = { "auc": 1}


def run_experiment(exp_file_path="exp.yml"):
    job_cls, job_func_str, EXPERIMENTS_DIR_PATH, WRK_DIR_PATH, exp_params = generate_params(exp_file_path=exp_file_path)
    job_cls = job_cls_map[job_cls]

    if not os.path.exists(EXPERIMENTS_DIR_PATH):
        os.makedirs(EXPERIMENTS_DIR_PATH)
    # save a copy of the exp.yml file to the experiment directory
    copyfile(exp_file_path, os.path.join(EXPERIMENTS_DIR_PATH, os.path.basename(exp_file_path)))

    exp_base_name = os.path.splitext(exp_file_path)[0]
    for i, params in enumerate(exp_params):
        EXPERIMENT_NAME = exp_base_name+"_"+str(i)
        OUTPUTS_DIR_PATH = os.path.join(EXPERIMENTS_DIR_PATH, EXPERIMENT_NAME)
        if not os.path.exists(OUTPUTS_DIR_PATH):
            os.makedirs(OUTPUTS_DIR_PATH)
        params_yml = os.path.join(OUTPUTS_DIR_PATH, "params.yml")
        with io.open(params_yml, 'w', encoding='utf8') as outfile:
            yaml.dump(params, outfile, default_flow_style=False, allow_unicode=True)

        job = job_cls(OUTPUTS_DIR_PATH=OUTPUTS_DIR_PATH)
        print("debug 1: {}".format(job_func_str))
        job_func = getattr(job, job_func_str)
        print("job params: {}".format(params))
        job_func(WRK_DIR_PATH=WRK_DIR_PATH, **params)


def generate_params(exp_file_path="exp.yml"):
    """generate the exp params combinations from the dictionary of mappings, with an optional list of choices for params
    exp"""
    exp = load_yaml(exp_file_path)
    exp = flatten(exp)
    job_cls = exp.pop("job_cls")
    job_func_str = exp.pop("job_func")
    EXPERIMENTS_DIR_PATH = exp.pop("EXPERIMENTS_DIR_PATH")
    WRK_DIR_PATH = exp.pop("WRK_DIR_PATH")
    num_files = exp.pop("num_files", None)
    
    fixed_params = dict()
    testing_params = dict()

    for k,v in exp.items():
        # params that are stored as list will be distributed
        # tuples, etc. will not be treated this way
        if isinstance(v, list):
            testing_params[k] = v
        else:
            fixed_params[k] = v

    # find all the hyper-parameter combinations
    testing_params = product_dict(**testing_params)

    # sample the number of experiments
    if num_files:
        testing_params = sample(testing_params, num_files)

    # update the parameter combinations with the fixed parameters
    params = [update(testing_params_exp, fixed_params) for testing_params_exp in testing_params]
    print("debug 0: {}".format(job_func_str))
    return job_cls, job_func_str, EXPERIMENTS_DIR_PATH, WRK_DIR_PATH, params


def load_yaml(yaml_file_path="params.yml"):
    # load the hyper-params
    with open(yaml_file_path, 'r') as stream:
        data = yaml.load(stream)
    return data

def analyze_exp(EXPERIMENTS_DIR_PATH, params_file_name="params.yml", exp_file_name="exp.yml",
                metric="auc", mof_metric="mad", file_char = "mof", round_arg=4, rt_jobs_score=None,
                rt_jobs_metric_interval=None):

    # define func for measure of fit
    if mof_metric == "mad":
        mof_func = robust.mad
    elif mof_metric == "std":
        mof_func = np.std

    glob_regex = os.path.join(EXPERIMENTS_DIR_PATH, "*", params_file_name)
    params_ymls = glob(glob_regex)
    job_results = defaultdict(lambda : defaultdict(float))
    metric_marg_scores = defaultdict(lambda : defaultdict(lambda : defaultdict(list)))

    exp_file_path = os.path.join(EXPERIMENTS_DIR_PATH, exp_file_name)
    exp = load_yaml(exp_file_path)
    exp = flatten(exp)
    n_metric_intervals = exp.pop("n_epochs")/exp.pop("metrics_epoch_freq")

    testing_params = dict()

    # find all the testing parameters
    for k,v in exp.items():
        # params that are stored as list will be distributed
        # tuples, etc. will not be treated this way
        if isinstance(v, list):
            testing_params[k] = v

    # collect the experiment parameters and testing parameters used in the experiment
    exp_params = {}
    testing_params_opts = defaultdict(set)

    for params_yml in params_ymls:
        JOB_PATH = os.path.dirname(params_yml)
        job_name = os.path.split(JOB_PATH)[1]
        params = load_yaml(params_yml)
        exp_params[job_name] = {}
        for k in testing_params.keys():
            exp_params[job_name][k] = json.dumps(params[k], sort_keys=True)
            testing_params_opts[k].add(json.dumps(params[k], sort_keys=True))

        # get the file with the combined metrics score
        if file_char == "mof" or file_char == "csv":
            print("debug job list: {}".format(os.listdir(JOB_PATH)))
            print("debug job list: {}".format([file for file in os.listdir(JOB_PATH) if file_char in file]))
            job_metrics_file = [file for file in os.listdir(JOB_PATH) if file_char in file][0]
        else:
            raise ValueError("file_char {} not recognized".format(file_char))

        JOB_METRICS_PATH = os.path.join(JOB_PATH, job_metrics_file)

        metric_col = metric_col_map[metric]
        p = re.compile(r'\d+\.\d+')

        # collect the job and marginal metric results
        with open(JOB_METRICS_PATH) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            next(csv_reader, None)
            for i, row in enumerate(csv_reader):
                metric_result = row[metric_col]
                metric_result = [float(result) for result in p.findall(metric_result)][0:2]
                job_results[i][job_name] = metric_result
                for k,v in exp_params[job_name].items():
                    param_name = str(k) + "." + str(v)
                    metric_marg_scores[i][param_name] = metric_marg_scores[i][param_name] + [metric_result[0]]

    # combine the marginal metric results
    for i in range(n_metric_intervals):
        for k,v in testing_params_opts.items():
            param_name = str(k) + "." + str(v)
            metric_marg_scores_list = metric_marg_scores[i][param_name]
            mean_result = np.mean(metric_marg_scores_list)
            mof_result = mof_func(metric_marg_scores_list)
            metric_marg_scores[i][param_name] = str(np.round(mean_result, round_arg)) + "+/-" + \
                                                str(np.round(mof_result, round_arg))

    # create the marginal hyperparameter results file
    hyp_metrics_log = "marg_hyp_log.csv"
    hyp_metrics_log_path = os.path.join(EXPERIMENTS_DIR_PATH, hyp_metrics_log)
    with open(hyp_metrics_log_path, "w") as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        writer.writerow([param_name for param_name in sorted(metric_marg_scores[0].keys())]+["mean", mof_metric])
        for i in range(n_metric_intervals):
            i_mean = np.mean(job_results[i].values())
            i_mof = mof_func(job_results[i].values())
            writer.writerow([metric_marg_scores[i][param_name] for param_name in sorted(metric_marg_scores[0].keys())]
                            + [i_mean, i_mof])

    # create the ranked marginal hyperparameter results file
    rank_hyp_metrics_log = "rank_marg_hyp_log.csv"
    rank_hyp_metrics_log_path = os.path.join(EXPERIMENTS_DIR_PATH, rank_hyp_metrics_log)
    with open(rank_hyp_metrics_log_path, "w") as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        for i in range(n_metric_intervals):
            ranked_metric_marg_scores = sorted(metric_marg_scores[i].items(), key = lambda x: x[1], reverse=True)
            i_mean = np.mean(job_results[i].values())
            i_mof = mof_func(job_results[i].values())
            writer.writerow([ranked_marg_score[0] for ranked_marg_score in ranked_metric_marg_scores] +
                            ["mean", mof_metric])
            writer.writerow([ranked_marg_score[1] + " % rank {:.1%}".format(float(i)/len(ranked_metric_marg_scores))
                             for i, ranked_marg_score in enumerate(ranked_metric_marg_scores)] + [i_mean, i_mof])

    rank_job_log = "rank_job_log.csv"
    rank_job_log_path = os.path.join(EXPERIMENTS_DIR_PATH, rank_job_log)
    rt_jobs = []
    with open(rank_job_log_path, "w") as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        for i in range(n_metric_intervals):
            ranked_job_results = sorted(job_results[i].items(), key = lambda x: x[1], reverse=True)
            i_mean = np.mean(job_results[i].values())
            i_mof = mof_func(job_results[i].values())
            writer.writerow([ranked_job_result[0] for ranked_job_result in ranked_job_results] +
                            ["mean", mof_metric])
            writer.writerow([ranked_job_result[1] + " % rank {:.1%}".format(float(i)/len(ranked_job_results))
                             for i, ranked_job_result in enumerate(ranked_job_results)] + [i_mean, i_mof])

            # filter the job_paths based on the job scores and other criteria
            if rt_jobs_score is not None and (rt_jobs_metric_interval is None or i == rt_jobs_metric_interval):
                for job_name, results in ranked_job_results:
                    if results >= rt_jobs_score:
                        if job_name not in rt_jobs:
                            rt_jobs.append(job_name)
    return rt_jobs








