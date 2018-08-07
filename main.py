from job.drive import DriveJob
import os

if __name__ == '__main__':
    experiments_dir_path = "/Users/arvind.m.vepa/vessel_seg_data/drive/experiments"
    experiment_name = "test"
    job = DriveJob(OUTPUTS_DIR_PATH=os.path.join(experiments_dir_path, experiment_name))
    job.run_single_model(WRK_DIR_PATH="/Users/arvind.m.vepa/vessel_seg_data/drive")
    print("done")