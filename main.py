from job.dsa import DsaJob
from job.drive import DriveJob, DriveCustomJob
from job.stare import StareJob
from job.chase import ChaseJob
import os
from params.params import run_experiment, analyze_exp
from params.misc import copy_exp

if __name__ == '__main__':
    exp_file_name = "exp_drive_job_.yml"
    run_experiment(exp_file_path=exp_file_name)
    EXPERIMENTS_DIR_PATH = r"C:\Users\arvin\dev\vessel-segmentation\sample_drive_exp"
    #TOP_EXPERIMENTS_DIR_PATH = r"C:\Users\arvin\dev\vessel-segmentation\copy_sample1"
    jobs = analyze_exp(EXPERIMENTS_DIR_PATH, exp_file_name=exp_file_name, file_char="csv", rt_jobs_score=.93)
    print(jobs)
    #copy_exp(jobs, EXPERIMENTS_DIR_PATH, TOP_EXPERIMENTS_DIR_PATH, exp_file_name)
    #analyze_exp(TOP_EXPERIMENTS_DIR_PATH, exp_file_name=exp_file_name, file_char="csv")
