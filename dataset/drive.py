
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


    def __init__(self, **kwargs):
        super(DriveDataset, self).__init__(**kwargs)
        

    @property
    def network_cls(self):
        return DriveNetwork


    

