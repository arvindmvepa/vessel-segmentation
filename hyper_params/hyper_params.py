import yaml
import io
from utilities.misc import flatten, product_dict, update
from random import sample
import os


def run_experiment(job_cls, job_func, EXPERIMENTS_DIR_PATH, WRK_DIR_PATH, exp_file="exp.yml", num_files=None, **kwargs):
    hyper_params_exps = generate_hyper_param_dict(exp_file=exp_file, num_files=num_files)
    exp_base_name = os.path.splitext(exp_file)[0]

    for i, hyper_params_exp in enumerate(hyper_params_exps):
        EXPERIMENT_NAME = exp_base_name+"_"+str(i)
        OUTPUTS_DIR_PATH = os.path.join(EXPERIMENTS_DIR_PATH, EXPERIMENT_NAME)
        hyper_params_yml = os.path.join(OUTPUTS_DIR_PATH, "hyper_params.yml")

        with io.open(hyper_params_yml, 'w', encoding='utf8') as outfile:
            yaml.dump(hyper_params_exp, outfile, default_flow_style=False, allow_unicode=True)

        job = job_cls(OUTPUTS_DIR_PATH=OUTPUTS_DIR_PATH)
        job_func = getattr(job, job_func)
        job_kwargs = update(hyper_params_exp, kwargs)
        job.job_func(WRK_DIR_PATH=WRK_DIR_PATH, **job_kwargs)


def generate_hyper_param_dict(exp_file="exp.yml", num_files=None):
    """generate the exp params combinations from the dictionary of mappings, with an optional list of choices for
    hyper-params exp"""

    with open(exp_file, 'r') as stream:
        exp = yaml.load(stream)

    exp = flatten(exp)
    exp_dict = dict()
    hyper_params = dict()

    for k,v in exp.items():
        # params that are stored as list will be distributed
        # tuples, etc. will not be treated this way
        if isinstance(v, list):
            hyper_params[k] = v
        else:
            exp_dict[k] = v

    # find all the hyper-parameter combinations
    hyper_params_exps = product_dict(hyper_params)

    # sample the number of experiments
    if num_files:
        hyper_params_exps = sample(hyper_params_exps, num_files)

    # update the hyper-parameter combinations with the fixed parameters
    hyper_params_exps = [update(hyper_params_exp, exp_dict) for hyper_params_exp in hyper_params_exps]

    return hyper_params_exps


def load_yaml(file="hyper_params.yml"):
    # load the hyper-params
    with open(file, 'r') as stream:
        data = yaml.load(stream)
    return data

def analyze_exp():


