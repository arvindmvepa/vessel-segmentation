import os
import numpy as np
from skimage import io as skio
import cv2
from PIL import Image

from dataset.base import Dataset
from network.drive import DriveNetwork
from utilities.image_preprocessing import preprocessing

class DriveDataset(Dataset):

    masks_dir = "masks"
    targets_dir = "targets"

    def __init__(self, batch_size=1, data_dir='drive', train_subdir="train", test_subdir="test", sgd=False):
        super(self, DriveDataset).__init__(batch_size=batch_size, data_dir=data_dir, train_subdir=train_subdir,
                                           test_subdir=test_subdir, sgd=sgd)

        self.train_images, self.train_masks, self.train_targets = self.train_data
        self.test_images, self.test_masks, self.test_targets = self.test_data

    @staticmethod
    def get_images_from_file(self, dir):
        dir_path = os.path.join(self.data_dir, dir)

        images = []
        masks = []
        targets = []

        images_dir = os.path.join(dir_path, self.images_folder)
        mask_dir = os.path.join(dir_path, self.masks_folder)
        target_dir = os.path.join(dir_path, self.targets_folder)

        image_files = sorted(os.listdir(images_dir))
        mask_files = sorted(os.listdir(mask_dir))
        target_files = sorted(os.listdir(target_dir))

        for image_file,mask_file,target_file in zip(image_files, mask_files, target_files):

            image_arr = cv2.imread(image_file, 1)
            image_arr = image_arr[:, :, 1]

            top_pad = int((DriveNetwork.FIT_IMAGE_HEIGHT - DriveNetwork.IMAGE_HEIGHT) / 2)
            bot_pad = (DriveNetwork.FIT_IMAGE_HEIGHT - DriveNetwork.IMAGE_HEIGHT) - top_pad
            left_pad = int((DriveNetwork.FIT_IMAGE_WIDTH - DriveNetwork.IMAGE_WIDTH) / 2)
            right_pad = (DriveNetwork.FIT_IMAGE_WIDTH - DriveNetwork.IMAGE_WIDTH) - left_pad

            image_arr = cv2.copyMakeBorder(image_arr, left_pad, right_pad, top_pad, bot_pad, cv2.BORDER_CONSTANT, 0)
            image_arr = np.multiply(image_arr, 1.0/255)
            images.append(image_arr)

            mask = Image.open(mask_file)
            mask_arr = np.array(mask)
            mask_arr = mask_arr / 255
            masks.append(mask_arr)

            target_arr = np.array(skio.imread(target_file))
            target_arr = cv2.threshold(target_arr, 127, 1, cv2.THRESH_BINARY)[1]
            targets.append(target_arr)

        return np.asarray(images), np.asarray(masks), np.asarray(targets)


    def next_batch(self):
        images = []
        masks = []
        targets = []

        if self.sgd:
            samples = np.random.choice(len(self.train_images), self.batch_size)

        for i in range(self.batch_size):
            if self.sgd:
                images.append(np.array(self.train_images[samples[i]]))
                masks.append(np.array(self.train_masks[samples[i]]))
                targets.append(np.array(self.train_targets[samples[i]]))
            else:
                images.append(np.array(self.train_images[self.pointer + i]))
                masks.append(np.array(self.train_masks[self.pointer + i]))
                targets.append(np.array(self.train_targets[self.pointer + i]))

        self.pointer += self.batch_size
        return np.array(images, dtype=np.uint8), np.array(masks, dtype=np.uint8), np.array(targets, dtype=np.uint8)

    def get_data_for_tensorflow(self, dataset="train"):
        if dataset == "train":
            return np.reshape(self.train_images, (self.train_images.shape[0], self.train_images.shape[1],
                                                  self.train_images.shape[2], 1)),\
                   np.reshape(self.train_masks, (self.train_masks.shape[0], self.train_masks.shape[1],
                                            self.train_masks.shape[2], 1)),\
                   np.reshape(self.train_targets, (self.train_targets.shape[0], self.train_targets.shape[1],
                                                   self.train_targets.shape[2], 1))
        if dataset == "test":
            return np.reshape(self.test_images, (self.test_images.shape[0], self.test_images.shape[1],
                                                  self.test_images.shape[2], 1)),\
                   np.reshape(self.test_masks, (self.test_masks.shape[0], self.test_masks.shape[1],
                                            self.test_masks.shape[2], 1)),\
                   np.reshape(self.test_targets, (self.test_targets.shape[0], self.test_targets.shape[1],
                                                   self.test_targets.shape[2], 1))

    def get_inverse_pos_freq(self, images, targets, masks):
        total_pos = 0
        total_num_pixels = 0
        for target, mask in zip(self.targets, self.masks):
            target = np.multiply(target, mask)
            total_pos += np.count_nonzero(target)
            total_num_pixels += np.count_nonzero(mask)
        total_neg = total_num_pixels - total_pos
        return total_neg/total_pos, float(total_neg)/float(total_num_pixels), float(total_pos)/float(total_num_pixels)

    @property
    def test_set(self):
        return np.array(self.test_images, dtype=np.uint8), np.array(self.test_masks, dtype=np.uint8), \
               np.array(self.test_targets, dtype=np.uint8)
