import numpy as np
import math
from math import ceil
from utilities.misc import find_class_balance

class Dataset:

    images_dir = "images"

    def __init__(self, batch_size=1, data_dir = ".", train_subdir="train", test_subdir="test", sgd = False):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.sgd = sgd

        self.train_data = self.get_images_from_file(train_subdir)
        self.test_data = self.get_images_from_file(test_subdir)

        self.pointer = 0

    def get_images_from_file(self,dir):
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

    def num_batches_in_epoch(self):
        return int(math.floor(len(self.train_data[0]) / self.batch_size))

    def reset_batch_pointer(self):
        self.pointer = 0

    def next_batch(self):
        raise NotImplementedError("Method Not Implemented")

    def get_tuned_pos_ce_weight(self, tuning_constant=1.0):
        return tuning_constant*self.get_inverse_pos_freq()[0]

    def get_inverse_pos_freq(self, images, targets, **kwargs):
        raise NotImplementedError("Method Not Implemented")

    def evaluate_on_test_set(self):
        raise NotImplementedError("Method Not Implemented")

    @property
    def test_set(self):
        raise ValueError("Property Not Definned")
