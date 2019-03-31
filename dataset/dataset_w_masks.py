import os
import numpy as np
from skimage import io as skio
import cv2
from PIL import Image

from dataset.base import Dataset
from utilities.image_preprocessing import preprocessing

class DatasetWMasks(Dataset):

    MASKS_DIR = "masks"
    TARGETS_DIR = "targets"

    def __init__(self, early_stopping=False, masks_provided=True, init_mask_imgs=False, mask_threshold = None, **kwargs):
        self.mask_provided = masks_provided
        self.init_mask_imgs = init_mask_imgs
        self.mask_threshold = mask_threshold
        super(DatasetWMasks, self).__init__(early_stopping=early_stopping, **kwargs)
        if early_stopping:
            self.train_images, self.train_masks, self.train_targets = self.train_data
            self.val_images, self.val_masks, self.val_targets = self.val_data
        else:
            self.train_images, self.train_masks, self.train_targets = self.train_data
        self.test_images, self.test_masks, self.test_targets = self.test_data


    def get_images_from_file(self, DIR_PATH, file_indices=None, hist_eq=None, clahe_kwargs=None, gamma=None,
                             per_image_z_score_norm=False, per_image_zero_center=False,
                             per_image_zero_center_scale=False):

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

        if self.mask_provided or self.init_mask_imgs:
            mask_files = sorted(os.listdir(MASKS_DIR_PATH))
            if file_indices is not None:
                mask_files = [mask_files[i] for i in file_indices]

        for i, (image_file,target_file) in enumerate(zip(image_files, target_files)):

            image_arr = cv2.imread(os.path.join(IMAGES_DIR_PATH,image_file), 1)
            grn_chnl_arr = image_arr[:, :, 1]

            top_pad = int((self.network_cls.FIT_IMAGE_HEIGHT - self.network_cls.IMAGE_HEIGHT) / 2)
            bot_pad = (self.network_cls.FIT_IMAGE_HEIGHT - self.network_cls.IMAGE_HEIGHT) - top_pad
            left_pad = int((self.network_cls.FIT_IMAGE_WIDTH - self.network_cls.IMAGE_WIDTH) / 2)
            right_pad = (self.network_cls.FIT_IMAGE_WIDTH - self.network_cls.IMAGE_WIDTH) - left_pad

            grn_chnl_arr = cv2.copyMakeBorder(grn_chnl_arr, top_pad, bot_pad, left_pad, right_pad, cv2.BORDER_CONSTANT, 0)
            # apply image pre-processing
            grn_chnl_arr = preprocessing(grn_chnl_arr, histo_eq=hist_eq, clahe_kwargs=clahe_kwargs, gamma=gamma,
                                         per_image_z_score_norm=per_image_z_score_norm,
                                         per_image_zero_center=per_image_zero_center,
                                         per_image_zero_center_scale=per_image_zero_center_scale)
            grn_chnl_arr = grn_chnl_arr * 1.0/255.0
            images.append(grn_chnl_arr)

            if self.mask_provided or self.init_mask_imgs:
                mask_file = mask_files[i]
                mask = Image.open(os.path.join(MASKS_DIR_PATH,mask_file))
                mask_arr = np.array(mask)
                mask_arr = mask_arr * 1.0 / 255.0
                # load base files to produce masks
                if self.init_mask_imgs:
                    # scale scores by 100
                    mask_arr = mask_arr * 100.0
                    mask_arr = np.where(mask_arr > self.mask_threshold, 1, 0)
            else:
                # convert from BGR to CLIELAB color space
                l_image_arr = cv2.cvtColor(image_arr, cv2.COLOR_BGR2LAB)[:,:,0]*(100.0/255.0)
                mask_arr = np.where(l_image_arr > self.mask_threshold, 1, 0.0)

            # apply morphological open operation to created masks
            if not self.mask_provided:
                kernel = np.ones((3, 3), np.uint8)
                mask_arr = cv2.morphologyEx(mask_arr.astype(np.uint8), cv2.MORPH_OPEN, kernel)

            masks.append(mask_arr)

            target_arr = np.array(skio.imread(os.path.join(TARGETS_DIR_PATH,target_file)))
            target_arr = np.where(target_arr > 127,1.0,0.0)

            targets.append(target_arr)

        return images, np.asarray(masks), np.asarray(targets)


    def next_batch(self):
        images = []
        masks = []
        targets = []

        if self.w_replacement:
            samples = np.random.choice(len(self.train_images), self.batch_size)

        for i in range(self.batch_size):
            if self.w_replacement:
                images.append(np.array(self.train_images[samples[i]]))
                masks.append(np.array(self.train_masks[samples[i]]))
                targets.append(np.array(self.train_targets[samples[i]]))
            else:
                images.append(np.array(self.train_images[self.pointer + i]))
                masks.append(np.array(self.train_masks[self.pointer + i]))
                targets.append(np.array(self.train_targets[self.pointer + i]))
        if self.seq is not None:
            seq_det = self.seq._to_deterministic()
            images = self.apply_aug(images, seq_det)
            masks = self.apply_aug(masks, seq_det, masks=True)
            targets = self.apply_aug(targets, seq_det, masks=True)
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