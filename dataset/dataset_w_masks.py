import os
import numpy as np
from skimage import io as skio
import cv2
from PIL import Image
from scipy.misc import imsave

from dataset.base import Dataset

class DatasetWMasks(Dataset):

    MASKS_DIR = "masks"
    TARGETS_DIR = "targets"

    def __init__(self, WRK_DIR_PATH, batch_size=1, TRAIN_SUBDIR="train", TEST_SUBDIR="test", sgd=True,
                 masks_provided=True, mask_threshold = None, cv_train_inds = None, cv_test_inds = None):
        self.mask_provided = masks_provided
        self.mask_threshold = mask_threshold
        super(DatasetWMasks, self).__init__(WRK_DIR_PATH=WRK_DIR_PATH, batch_size=batch_size, TRAIN_SUBDIR=TRAIN_SUBDIR,
                                           TEST_SUBDIR=TEST_SUBDIR, sgd=sgd, cv_train_inds=cv_train_inds,
                                            cv_test_inds=cv_test_inds)

        self.train_images, self.train_masks, self.train_targets = self.train_data
        self.test_images, self.test_masks, self.test_targets = self.test_data


    def get_images_from_file(self, DIR_PATH, file_indices=None):

        images = []
        masks = []
        targets = []

        IMAGES_DIR_PATH = os.path.join(DIR_PATH, self.IMAGES_DIR)
        MASKS_DIR_PATH = os.path.join(DIR_PATH, self.MASKS_DIR)
        TARGETS_DIR_PATH = os.path.join(DIR_PATH, self.TARGETS_DIR)

        image_files = sorted(os.listdir(IMAGES_DIR_PATH))
        target_files = sorted(os.listdir(TARGETS_DIR_PATH))

        if file_indices is not None:
            image_files = [image_files[i] for i in file_indices]
            target_files = [target_files[i] for i in file_indices]

        if self.mask_provided:
            mask_files = sorted(os.listdir(MASKS_DIR_PATH))
            if file_indices is not None:
                mask_files = [mask_files[i] for i in file_indices]

        for i, (image_file,target_file) in enumerate(zip(image_files, target_files)):

            image_arr = cv2.imread(os.path.join(IMAGES_DIR_PATH,image_file), 1)
            grn_image_arr = image_arr[:, :, 1]

            top_pad = int((self.network_cls.FIT_IMAGE_HEIGHT - self.network_cls.IMAGE_HEIGHT) / 2)
            bot_pad = (self.network_cls.FIT_IMAGE_HEIGHT - self.network_cls.IMAGE_HEIGHT) - top_pad
            left_pad = int((self.network_cls.FIT_IMAGE_WIDTH - self.network_cls.IMAGE_WIDTH) / 2)
            right_pad = (self.network_cls.FIT_IMAGE_WIDTH - self.network_cls.IMAGE_WIDTH) - left_pad

            grn_image_arr = cv2.copyMakeBorder(grn_image_arr, top_pad, bot_pad, left_pad, right_pad, cv2.BORDER_CONSTANT, 0)
            grn_image_arr = grn_image_arr * 1.0/255.0
            images.append(grn_image_arr)

            if self.mask_provided:
                mask_file = mask_files[i]
                mask = Image.open(os.path.join(MASKS_DIR_PATH,mask_file))
                mask_arr = np.array(mask)
                mask_arr = mask_arr * 1.0 / 255.0
            else:
                l_image_arr = cv2.cvtColor(image_arr, cv2.COLOR_BGR2LAB)[:,:,0]*(100.0/255.0)
                mask_arr = np.where(l_image_arr > self.mask_threshold,1.0,0.0)
            masks.append(mask_arr)

            target_arr = np.array(skio.imread(os.path.join(TARGETS_DIR_PATH,target_file)))
            target_arr = np.where(target_arr > 127,1.0,0.0)

            targets.append(target_arr)

            orig_img = image_file
            orig_pth = os.path.join(self.WRK_DIR_PATH, orig_img)
            imsave(orig_pth, grn_image_arr * 255.0)
            target_img = "target_" + image_file
            target_pth = os.path.join(self.WRK_DIR_PATH, target_img)
            imsave(target_pth, target_arr * 255.0)

            mask_img = "mask_" + image_file
            mask_pth = os.path.join(self.WRK_DIR_PATH, mask_img)
            imsave(mask_pth, mask_arr * 255.0)

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
        return np.array(images), np.array(masks), np.array(targets)

    def get_inverse_pos_freq(self, masks, targets):
        total_pos = 0
        total_num_pixels = 0
        for target, mask in zip(targets, masks):
            target = np.multiply(target, mask)
            total_pos += np.count_nonzero(target)
            total_num_pixels += np.count_nonzero(mask)
        total_neg = total_num_pixels - total_pos
        return float(total_neg)/float(total_pos), float(total_neg)/float(total_num_pixels), \
               float(total_pos)/float(total_num_pixels)

    @property
    def test_set(self):
        return np.array(self.test_images), np.array(self.test_masks), np.array(self.test_targets)