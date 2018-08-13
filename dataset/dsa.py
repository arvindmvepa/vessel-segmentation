import os
import numpy as np
from skimage import io as skio
import cv2

from dataset.base import Dataset
from network.dsa import DsaNetwork

class DsaDataset(Dataset):

    TARGETS1_DIR = "targets1"
    TARGETS2_DIR = "targets2"

    def __init__(self, batch_size=1, WRK_DIR_PATH ="./dsa", TRAIN_SUBDIR="train", TEST_SUBDIR="test", sgd = True,
                 cv_train_inds = None, cv_test_inds = None):
        super(DsaDataset, self).__init__(batch_size=batch_size, WRK_DIR_PATH=WRK_DIR_PATH, TRAIN_SUBDIR=TRAIN_SUBDIR,
                                         TEST_SUBDIR=TEST_SUBDIR, sgd=sgd, cv_train_inds=cv_train_inds,
                                         cv_test_inds=cv_test_inds)

        self.train_images, self.train_targets = self.train_data
        self.test_images, self.test_targets = self.test_data

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

    def next_batch(self):
        images = []
        targets = []

        if self.sgd:
            samples = np.random.choice(len(self.train_images), self.batch_size)

        for i in range(self.batch_size):
            if self.sgd:
                images.append(np.array(self.train_images[samples[i]]))
                targets.append(np.array(self.train_targets[samples[i]]))
            else:
                images.append(np.array(self.train_images[self.pointer + i]))
                targets.append(np.array(self.train_targets[self.pointer + i]))

        self.pointer += self.batch_size
        return np.array(images), np.array(targets)

    def get_data_for_tensorflow(self, dataset="train"):
        if dataset == "train":
            return np.reshape(self.train_images, (self.train_images.shape[0], self.train_images.shape[1],
                                                  self.train_images.shape[2], 1)),\
                   np.reshape(self.train_targets, (self.train_targets.shape[0], self.train_targets.shape[1],
                                                   self.train_targets.shape[2], 1))
        if dataset == "test":
            return np.reshape(self.test_images, (self.test_images.shape[0], self.test_images.shape[1],
                                                  self.test_images.shape[2], 1)),\
                   np.reshape(self.test_targets, (self.test_targets.shape[0], self.test_targets.shape[1],
                                                   self.test_targets.shape[2], 1))

    def get_inverse_pos_freq(self, targets):
        total_pos = 0
        total_num_pixels = 0
        for target in targets:
            total_pos += np.count_nonzero(target)
            total_num_pixels += DsaNetwork.IMAGE_WIDTH * DsaNetwork.IMAGE_HEIGHT
        total_neg = total_num_pixels - total_pos
        return float(total_neg)/float(total_pos), float(total_neg)/float(total_num_pixels), float(total_pos)/float(total_num_pixels)

    @property
    def test_set(self):
        return np.array(self.train_images), np.array(self.test_targets)