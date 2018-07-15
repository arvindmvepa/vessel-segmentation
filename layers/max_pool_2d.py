import tensorflow as tf

from layers.base import Layer
from utilities.layer_ops import get_incoming_shape, max_pool_2d, upsample_2d

class MaxPool2d(Layer):
    def __init__(self, kernel_size, name, skip_connection=False):
        self.kernel_size = kernel_size
        self.name = name
        self.skip_connection = skip_connection

    def create_layer(self, input):
        self.input_shape = get_incoming_shape(input)
        print(self.input_shape)
        return max_pool_2d(input, self.kernel_size)

    def create_layer_reversed(self, input, prev_layer=None, **kwargs):
        if self.skip_connection:
            input = tf.add(input, prev_layer)

        return upsample_2d(input, self.kernel_size)

    def get_description(self):
        return "M{}".format(self.kernel_size)