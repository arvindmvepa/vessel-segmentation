from job.dsa import DsaJob
from job.drive import DriveJob, DriveCustomJob
from job.stare import StareJob
from job.chase import ChaseJob
import os
from params.params import run_experiment

if __name__ == '__main__':
    run_experiment(exp_file="sample.yml")
