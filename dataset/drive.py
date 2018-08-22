
"""This is the file for the DriveDataset subclass"""

import os
import numpy as np
from skimage import io as skio
import cv2
from PIL import Image
from utilities.image_preprocessing import preprocessing


from dataset.dataset_w_masks import DatasetWMasks
from network.drive import DriveNetwork

class DriveDataset(DatasetWMasks):


    def __init__(self, batch_size=1, WRK_DIR_PATH='./drive', TRAIN_SUBDIR="train/", TEST_SUBDIR="test", sgd=True,
                 cv_train_inds = None, cv_test_inds = None, he_flag=False,clahe_flag=False,normalized_flag=False,gamma_flag=False, **kwargs):
        self.he_flag=he_flag
        self.clahe_flag=clahe_flag
        self.normalized_flag=normalized_flag
        self.gamma_flag=gamma_flag
        super(DriveDataset, self).__init__(batch_size=batch_size, WRK_DIR_PATH=WRK_DIR_PATH, TRAIN_SUBDIR=TRAIN_SUBDIR,
                                           TEST_SUBDIR=TEST_SUBDIR, sgd=sgd, cv_train_inds=cv_train_inds,
                                           cv_test_inds=cv_test_inds,**kwargs)
        

    @property

    def network_cls(self):
        return DriveNetwork


    

