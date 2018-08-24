import os
import numpy as np
from skimage import io as skio
import cv2

from dataset.base import Dataset
from network.chase import ChaseNetwork

class ChaseDataset(Dataset):

    TARGETS_DIR = "targets"

    def __init__(self, batch_size=1, WRK_DIR_PATH ="./dsa", TRAIN_SUBDIR="train", TEST_SUBDIR="test", sgd = True,
                 cv_train_inds = None, cv_test_inds = None):
        super(ChaseDataset, self).__init__(batch_size=batch_size, WRK_DIR_PATH=WRK_DIR_PATH, TRAIN_SUBDIR=TRAIN_SUBDIR,
                                         TEST_SUBDIR=TEST_SUBDIR, sgd=sgd, cv_train_inds=cv_train_inds,
                                         cv_test_inds=cv_test_inds)

        self.train_images, self.train_targets = self.train_data
        self.test_images, self.test_targets = self.test_data

    def get_images_from_file(self, DIR_PATH, file_indices=None):
        images = []
        targets = []

        IMAGES_DIR_PATH = os.path.join(DIR_PATH, self.IMAGES_DIR)
        TARGETS_DIR_PATH = os.path.join(DIR_PATH, self.TARGETS_DIR)

        image_files = sorted(os.listdir(IMAGES_DIR_PATH))
        target_files = sorted(os.listdir(TARGETS_DIR_PATH))

        if file_indices is not None:
            image_files = [image_files[i] for i in file_indices]
            target_files = [target_files[i] for i in file_indices]

        for image_file,target_file in zip(image_files, target_files):
            image_arr = cv2.imread(os.path.join(IMAGES_DIR_PATH, image_file), 1)
            image_arr = image_arr[:, :, 1]

            top_pad = int((ChaseNetwork.FIT_IMAGE_HEIGHT - ChaseNetwork.IMAGE_HEIGHT) / 2)
            bot_pad = (ChaseNetwork.FIT_IMAGE_HEIGHT - ChaseNetwork.IMAGE_HEIGHT) - top_pad
            left_pad = int((ChaseNetwork.FIT_IMAGE_WIDTH - ChaseNetwork.IMAGE_WIDTH) / 2)
            right_pad = (ChaseNetwork.FIT_IMAGE_WIDTH - ChaseNetwork.IMAGE_WIDTH) - left_pad

            image_arr = cv2.copyMakeBorder(image_arr, top_pad, bot_pad, left_pad, right_pad, cv2.BORDER_CONSTANT, 0)
            image_arr = image_arr * 1.0/255.0
            images.append(image_arr)

            target_arr = np.array(skio.imread(os.path.join(TARGETS_DIR_PATH,target_file)))
            target_arr = np.where(target_arr > 127,1.0,0.0)

            targets.append(target_arr)
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
            total_num_pixels += ChaseNetwork.IMAGE_WIDTH * ChaseNetwork.IMAGE_HEIGHT
        total_neg = total_num_pixels - total_pos
        return float(total_neg)/float(total_pos), float(total_neg)/float(total_num_pixels), float(total_pos)/float(total_num_pixels)

    @property
    def test_set(self):
        return np.array(self.train_images), np.array(self.test_targets)
