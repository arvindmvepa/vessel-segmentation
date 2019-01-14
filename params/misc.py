import os
from utilities.misc import copy_stuff

def copy_exp(jobs, EXPERIMENTS_DIR_PATH, TOP_EXPERIMENTS_DIR_PATH, exp_file_name="exp.yml"):
    if not os.path.exists(TOP_EXPERIMENTS_DIR_PATH):
        os.makedirs(TOP_EXPERIMENTS_DIR_PATH)
    for job in jobs:
        cur_loc = os.path.join(EXPERIMENTS_DIR_PATH, job)
        new_loc = os.path.join(TOP_EXPERIMENTS_DIR_PATH, job)
        copy_stuff(cur_loc, new_loc)
    cur_loc = os.path.join(EXPERIMENTS_DIR_PATH, exp_file_name)
    new_loc = os.path.join(TOP_EXPERIMENTS_DIR_PATH, exp_file_name)
    copy_stuff(cur_loc, new_loc)