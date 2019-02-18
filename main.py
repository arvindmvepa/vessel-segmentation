from params.params import run_experiment, analyze_exp, load_yaml
from utilities.misc import flatten
from params.misc import copy_exp

if __name__ == '__main__':
    exp_file_name = "exp_drive_job_conv_convt_1.yml"
    run_experiment(exp_file_path=exp_file_name, init_count=0)
    TOP_EXPERIMENTS_DIR_PATH = r"/root/vessel-segmentation1/top_drive_jobs"
    jobs = analyze_exp(exp_file_path=exp_file_name, file_char="csv", rt_jobs_score=.96)
    print(jobs)

    exp = load_yaml(exp_file_name)
    exp = flatten(exp)
    EXPERIMENTS_DIR_PATH = exp.pop("EXPERIMENTS_DIR_PATH")

    copy_exp(jobs, EXPERIMENTS_DIR_PATH, TOP_EXPERIMENTS_DIR_PATH, exp_file_name)
    analyze_exp(TOP_EXPERIMENTS_DIR_PATH, exp_file_path=exp_file_name, file_char="csv")