import tensorflow as tf

from layers.base import Layer
from utilities.layer_ops import get_incoming_shape, max_pool_2d, upsample_2d

class MaxPool2d(Layer):

    def __init__(self, kernel_size, name, add_to_input=False):
        self.kernel_size = kernel_size
        self.name = name
        self.add_to_input = add_to_input

    def create_layer(self, input, add_w_input=None, **kwargs):
        if self.add_to_input is not None:
            input = tf.add(input, add_w_input)
        self.input_shape = get_incoming_shape(input)
        print(self.input_shape)
        return self.apply_pool(input, self.kernel_size)

    def apply_pool(self, input, kernel_size):
        return max_pool_2d(input, kernel_size)

    def get_description(self):
        return "P{}".format(self.kernel_size)

class UnPool2d(Layer):

    def apply_pool(self, input, kernel_size):
        return upsample_2d(input, kernel_size)


