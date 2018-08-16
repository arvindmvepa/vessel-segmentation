import os
import numpy as np
from skimage import io as skio
import cv2
from PIL import Image

from dataset.base import Dataset
from network.chase import ChaseNetwork

class ChaseDataset(Dataset):

    targets_dir = "targets"

    def __init__(self, batch_size=1, WRK_DIR_PATH ="./chase", TRAIN_SUBDIR="train", TEST_SUBDIR="test", sgd = False):
        super(ChaseDataset, self).__init__(batch_size=batch_size, WRK_DIR_PATH=WRK_DIR_PATH, TRAIN_SUBDIR=TRAIN_SUBDIR,
                                         TEST_SUBDIR=TEST_SUBDIR, sgd=sgd)

        self.train_images, self.train_targets = self.train_data
        self.test_images, self.test_targets = self.test_data

    def get_images_from_file(self, dir):
        dir_path = os.path.join(self.WRK_DIR_PATH, dir)
        images = []
        targets = []

        images_dir = os.path.join(dir_path, self.IMAGES_DIR)
        image_files = sorted(os.listdir(images_dir))

        for file in image_files:
            image_file = os.path.join(images_dir, file)
            image = Image.open(image_file)
            image_arr = np.array(image)
            image_arr = np.multiply(image_arr, 1.0/255)
            images.append(image_arr)

            if os.path.exists(os.path.join(dir_path, self.targets_dir, file)):
                target_file = os.path.join(dir_path, self.targets_dir, file)
                target_arr = np.array(skio.imread(target_file))
                target_arr = np.where(target_arr > 127,1,0)
                targets.append(target_arr)
            else:
                raise ValueError("Path for target file for \'{}\' not defined".format(file))

            print("successfully loading")
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
        return np.array(images, dtype=np.uint8), np.array(targets, dtype=np.uint8)

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

    def get_inverse_pos_freq(self):
        total_pos = 0
        total_num_pixels = 0
        for target in self.train_targets:
            total_pos += np.count_nonzero(target)
            total_num_pixels += ChaseNetwork.IMAGE_WIDTH * ChaseNetwork.IMAGE_HEIGHT
        total_neg = total_num_pixels - total_pos
        return total_neg/total_pos, float(total_neg)/float(total_num_pixels), float(total_pos)/float(total_num_pixels)

    @property
    def test_set(self):
        return np.array(self.train_images, dtype=np.uint8), np.array(self.test_targets, dtype=np.uint8)
