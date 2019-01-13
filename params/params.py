import yaml
import io
from utilities.misc import flatten, product_dict, update
from shutil import copyfile
from random import sample
import os
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


def run_experiment(exp_file_path="exp.yml"):
    job_cls, job_func, EXPERIMENTS_DIR_PATH, WRK_DIR_PATH, exp_params = generate_params(exp_file_path=exp_file_path)
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
        job_func = getattr(job, job_func)
        print("debug 1: {}".format(job_func))
        job.job_func(WRK_DIR_PATH=WRK_DIR_PATH, **params)


def generate_params(exp_file_path="exp.yml"):
    """generate the exp params combinations from the dictionary of mappings, with an optional list of choices for params
    exp"""

    with open(exp_file_path, 'r') as stream:
        exp = yaml.load(stream)

    exp = flatten(exp)
    job_cls = exp.pop("job_cls")
    job_func = exp.pop("job_func")
    print("debug 0: {}".format(job_func))
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

    return job_cls, job_func, EXPERIMENTS_DIR_PATH, WRK_DIR_PATH, params


def load_yaml(yaml_file="params.yml"):
    # load the hyper-params
    with open(yaml_file, 'r') as stream:
        data = yaml.load(stream)
    return data

#def analyze_exp():


