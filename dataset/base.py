import numpy as np
import math
from math import ceil
from utilities.misc import find_class_balance
import os

class Dataset(object):

    IMAGES_DIR = "images"

    def __init__(self, batch_size=1, WRK_DIR_PATH =".", TRAIN_SUBDIR="train", TEST_SUBDIR="test", sgd = True,
                 cv_train_inds = None, cv_test_inds = None):
        self.WRK_DIR_PATH = WRK_DIR_PATH
        self.batch_size = batch_size
        self.sgd = sgd

        self.TRAIN_DIR_PATH = os.path.join(self.WRK_DIR_PATH, TRAIN_SUBDIR)
        if cv_test_inds is not None:
            self.TEST_DIR_PATH = os.path.join(self.WRK_DIR_PATH, TRAIN_SUBDIR)
        else:
            self.TEST_DIR_PATH = os.path.join(self.WRK_DIR_PATH, TEST_SUBDIR)

        self.train_data = self.get_images_from_file(self.TRAIN_DIR_PATH, cv_train_inds)
        self.test_data = self.get_images_from_file(self.TEST_DIR_PATH, cv_test_inds)

        self.pointer = 0

    def get_images_from_file(self, DIR_PATH, file_indices=None):
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
    def test_set(self):
        raise ValueError("Property Not Definned")
