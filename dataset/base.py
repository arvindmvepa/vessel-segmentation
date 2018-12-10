"""This is the file for the Dataset base class"""
import numpy as np
import math
from math import ceil
import os
from sklearn.model_selection import train_test_split
from imgaug import imgaug
from utilities.augmentation import apply_image_aug
from utilities.image_preprocessing import apply_normalization


class Dataset(object):

    IMAGES_DIR = "images"

    def __init__(self, batch_size=1, WRK_DIR_PATH =".", TRAIN_SUBDIR="train", TEST_SUBDIR="test", early_stopping=False,
                 early_stopping_val_prop = .1, sgd = True, cv_train_inds = None, cv_test_inds = None, seq = None,
                 hist_eq=None, clahe_kwargs=None, gamma=None, zero_center=False, per_image_z_score_norm=False,
                 per_image_zero_center=False, per_image_zero_center_scale=False, zero_center_scale=False,
                 z_score_norm=False, **kwargs):

        self.WRK_DIR_PATH = WRK_DIR_PATH
        self.batch_size = batch_size
        self.sgd = sgd

        self.TRAIN_DIR_PATH = os.path.join(self.WRK_DIR_PATH, TRAIN_SUBDIR)
        if cv_test_inds is not None:
            self.TEST_DIR_PATH = os.path.join(self.WRK_DIR_PATH, TRAIN_SUBDIR)
        else:
            self.TEST_DIR_PATH = os.path.join(self.WRK_DIR_PATH, TEST_SUBDIR)
        self.seq = seq

        self.train_data = self.get_images_from_file(self.TRAIN_DIR_PATH, cv_train_inds, hist_eq=hist_eq,
                                                    clahe_kwargs=clahe_kwargs, gamma=gamma,
                                                    per_image_z_score_norm=per_image_z_score_norm,
                                                    per_image_zero_center=per_image_zero_center,
                                                    per_image_zero_center_scale=per_image_zero_center_scale)
        self.train_data = list(self.train_data)
        print("debug")
        print(self.train_data)
        print("debug end")
        self.train_data[0], train_params = apply_normalization(self.train_data[0], zero_center=zero_center,
                                                               zero_center_scale=zero_center_scale,
                                                               z_score_norm=z_score_norm)

        if early_stopping:
            data = train_test_split(*self.train_data, test_size=early_stopping_val_prop)
            self.train_data = data[::2]
            self.val_data = data[1::2]
        self.test_data = self.get_images_from_file(self.TEST_DIR_PATH, cv_test_inds, hist_eq=hist_eq,
                                                   clahe_kwargs=clahe_kwargs, gamma=gamma,
                                                   per_image_z_score_norm=per_image_z_score_norm,
                                                   per_image_zero_center=per_image_zero_center,
                                                   per_image_zero_center_scale=per_image_zero_center_scale)
        self.test_data = list(self.test_data)
        self.test_data[0], _ = apply_normalization(self.test_data[0], zero_center=zero_center,
                                                   zero_center_scale=zero_center_scale, z_score_norm=z_score_norm,
                                                   train_params=train_params)

        self.pointer = 0

    def get_images_from_file(self, DIR_PATH, file_indices=None, hist_eq=None, clahe_kwargs=None,
                             per_image_normalization=False, gamma=None):
        raise NotImplementedError("Method Not Implemented")

    def get_data_for_tensorflow(self, dataset="train"):
        raise NotImplementedError("Method Not Implemented")

    def train_valid_test_split(self, X, ratio=None):
        if ratio is None:
            ratio = (0.5, .25, .25)

        N = len(X)
        return (
            X[:int(ceil(N * ratio[0]))],
            X[int(ceil(N * ratio[0])): int(ceil(N * ratio[0] + N * ratio[1]))],
            X[int(ceil(N * ratio[0] + N * ratio[1])):]
        )
    @staticmethod
    def tf_reshape(arrs):
        reshaped_arrs = []
        for arr in arrs:
            if len(arr.shape) == 2:
                reshaped_arrs += [np.reshape(arr, (1, arr.shape[0], arr.shape[1], 1))]
            if len(arr.shape) == 3:
                reshaped_arrs += [np.reshape(arr, (arr.shape[0], arr.shape[1], arr.shape[2], 1))]
        return tuple(reshaped_arrs)

    def apply_img_normalization(self, images):
        return np.array([(image-np.min(image))*1.0/(np.max(image)-np.min(image)) for image in images])

    def apply_aug(self, *args, **kwargs):
        return apply_image_aug(*args, **kwargs)

    def num_batches_in_epoch(self):
        return int(math.floor(len(self.train_data[0]) / self.batch_size))

    def reset_batch_pointer(self):
        self.pointer = 0

    def next_batch(self):
        raise NotImplementedError("Method Not Implemented")

    def get_tuned_pos_ce_weight(self, tuning_constant=1.0, *args):
        return tuning_constant*self.get_inverse_pos_freq(*args)[0]

    def get_inverse_pos_freq(self, images, targets, **kwargs):
        raise NotImplementedError("Method Not Implemented")

    def evaluate_on_test_set(self):
        raise NotImplementedError("Method Not Implemented")

    @property
    def network_cls(self):
        raise ValueError("Property Not Defined")

    @property
    def test_set(self):
        raise ValueError("Property Not Defined")
