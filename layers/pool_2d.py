import tensorflow as tf

from layers.base import Layer
from utilities.layer_ops import get_incoming_shape, pool_2d, upsample_2d

class Pool2d(Layer):

    def __init__(self, kernel_size, name, skip_connection=False):
        self.kernel_size = kernel_size
        self.name = name
        self.skip_connection = skip_connection

    def create_layer(self, input, pooling_method="MAX", center=False, **kwargs):
        self.input_shape = get_incoming_shape(input)
        print(self.input_shape)
        output = pool_2d(input, self.kernel_size, method=pooling_method)

        # zero-center activations
        if center:
            output = output - tf.reduce_mean(output)
        return output

    def create_layer_reversed(self, input, prev_layer=None, unpooling_method="nearest_neighbor", center=False,
                              **kwargs):
        if self.skip_connection:
            input = tf.add(input, prev_layer)
        output = upsample_2d(input, self.kernel_size, method=unpooling_method)

        # zero-center activations
        if center:
            output = output - tf.reduce_mean(output)
        return output

    def get_description(self):
        return "M{}".format(self.kernel_size)
