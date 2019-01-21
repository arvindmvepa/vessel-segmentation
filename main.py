#from job.dsa import DsaJob
#from job.drive import DriveJob, DriveCustomJob
#from job.stare import StareJob
#from job.chase import ChaseJob
import os
from params.params import run_experiment, analyze_exp, run_job, load_yaml
from utilities.misc import flatten
from params.misc import copy_exp

if __name__ == '__main__':
    exp_file_name = "exp_custom_drive_job.yml"
    #exp_file_name = "exp_drive_job.yml"
    #EXPERIMENTS_DIR_PATH = r"/Users/arvind.m.vepa/Documents/vessel segmentation/drive_exp"
    EXPERIMENTS_DIR_PATH = r"/Users/arvind.m.vepa/Documents/vessel segmentation/custom_drive_exp"
    exp_file_path = os.path.join(EXPERIMENTS_DIR_PATH, exp_file_name)

    #run_experiment(exp_file_path=exp_file_name, init_count=0)
    jobs = analyze_exp(EXPERIMENTS_DIR_PATH=EXPERIMENTS_DIR_PATH, exp_file_path=exp_file_path, file_char="csv",
                       rt_jobs_score=.94)
    print(jobs)

    #exp = load_yaml(exp_file_name)
    #exp = flatten(exp)
    #EXPERIMENTS_DIR_PATH = exp.pop("EXPERIMENTS_DIR_PATH")

    #TOP_EXPERIMENTS_DIR_PATH = r"/Users/arvind.m.vepa/Documents/vessel segmentation/top_drive_jobs"
    TOP_EXPERIMENTS_DIR_PATH = r"/Users/arvind.m.vepa/Documents/vessel segmentation/top_custom_drive_jobs"

    copy_exp(jobs, EXPERIMENTS_DIR_PATH, TOP_EXPERIMENTS_DIR_PATH, exp_file_name)
    analyze_exp(EXPERIMENTS_DIR_PATH=TOP_EXPERIMENTS_DIR_PATH, exp_file_path=exp_file_path, file_char="csv")
