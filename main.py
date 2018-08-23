from job.drive import DriveJob
from job.dsa import DsaJob
import os

if __name__ == '__main__':
    EXPERIMENTS_DIR_PATH = "c:/vessel-segmentation/experiments"
    EXPERIMENT_NAME = "/drive_example_/"
    Layer_param={'conv_1_1_ks':3,'conv_1_1_oc':64,'conv_1_1_dilation':1,
            'max_1_ks':2,
            'conv_2_1_ks':3,'conv_2_1_oc':128,'conv_2_1_dilation':1,
            'max_2_ks':2,
            'conv_3_1_ks':3,'conv_3_1_oc':256,'conv_3_1_dilation':1,
            'conv_3_2_ks':3,'conv_3_2_oc':256,'conv_3_2_dilation':2,
            'max_3_ks':2,
            'conv_4_1_ks':7,'conv_4_1_oc':4096,'conv_4_1_dilation':1,
            'conv_4_2_ks':1,'conv_4_2_oc':4096,'conv_4_2_dilation':1
            }
            
    job = DriveJob(OUTPUTS_DIR_PATH=os.path.join(EXPERIMENTS_DIR_PATH, EXPERIMENT_NAME))
    job.run_cross_validation(WRK_DIR_PATH="c:/vessel-segmentation/DRIVE",metrics_epoch_freq=1,
            viz_layer_epoch_freq=1,n_epochs=5,gpu_device='/gpu:1', weight_init='He',regularizer='L2',Relu=False,
            learningrate=0.001, Beta1=0.9,Beta2=0.999,epsilon=10**-8,Layer_param=Layer_param, he_flag=False, clahe_flag=            False, normalized_flag=False, gamma_flag=False,keep_prob=1 )
