from job.dsa import DsaJob
from job.drive import DriveJob, DriveCustomJob
from job.stare import StareJob
from job.chase import ChaseJob
import os
from params.params import run_experiment, analyze_exp

if __name__ == '__main__':
    #run_experiment(exp_file_path="sample.yml")
    analyze_exp(r"C:\Users\arvin\dev\vessel-segmentation\exp_sample", exp_file_name="sample.yml")
