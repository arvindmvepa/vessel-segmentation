import os
import numpy as np
from skimage import io as skio
import cv2

from dataset.dataset_wo_masks import DatasetWoMasks
from network.dsa import DsaNetwork

class DsaDataset(DatasetWoMasks):

    TARGETS1_DIR = "targets1"
    TARGETS2_DIR = "targets2"

    def __init__(self, batch_size=1, WRK_DIR_PATH ="./dsa", TRAIN_SUBDIR="train", TEST_SUBDIR="test", sgd = True,
                 cv_train_inds = None, cv_test_inds = None):
        super(DsaDataset, self).__init__(batch_size=batch_size, WRK_DIR_PATH=WRK_DIR_PATH, TRAIN_SUBDIR=TRAIN_SUBDIR,
                                         TEST_SUBDIR=TEST_SUBDIR, sgd=sgd, cv_train_inds=cv_train_inds,
                                         cv_test_inds=cv_test_inds)

    def get_images_from_file(self, DIR_PATH, file_indices=None):
        images = []
        targets = []

        IMAGES_DIR_PATH = os.path.join(DIR_PATH, self.IMAGES_DIR)

        image_files = sorted(os.listdir(IMAGES_DIR_PATH))
        if file_indices is not None:
            image_files = [image_files[i] for i in file_indices]

        for image_file in image_files:
            image_path = os.path.join(IMAGES_DIR_PATH, image_file)
            image_arr = cv2.imread(image_path, 0)
            image_arr = np.multiply(image_arr, 1.0 / 255)
            images.append(image_arr)

            if os.path.exists(os.path.join(DIR_PATH, self.TARGETS1_DIR, image_file)):
                target_file = os.path.join(DIR_PATH, self.TARGETS1_DIR, image_file)
                target_arr = np.array(skio.imread(target_file))
                target_arr = np.where(target_arr > 127,1,0)
                targets.append(target_arr)
            elif os.path.exists(os.path.join(DIR_PATH, self.TARGETS2_DIR, image_file)):
                target_file = os.path.join(DIR_PATH, self.TARGETS2_DIR, image_file)
                target_arr = np.array(skio.imread(target_file))[:, :, 3]
                target_arr = np.where(target_arr > 127,1,0)
                targets.append(target_arr)
            else:
                raise ValueError("Path for target file for \'{}\' not defined".format(image_file))

        return np.asarray(images), np.asarray(targets)

    @property
    def network_cls(self):
        return DsaNetwork