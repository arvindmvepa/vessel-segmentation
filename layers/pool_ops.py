import tensorflow as tf

from layers.base import Layer
from utilities.layer_ops import get_incoming_shape, pool_2d, upsample_2d


class Pool(Layer):

    def __init__(self, kernel_size, name, add_to_input=False, **kwargs):
        super(Pool, self).__init__(**kwargs)
        self.kernel_size = kernel_size
        self.name = name
        self.add_to_input = add_to_input

    def create_layer(self, input, add_w_input=None, pooling_method="MAX", unpooling_method="nearest_neighbor",
                     center=False, **kwargs):
        self.input_shape = get_incoming_shape(input)
        print(self.input_shape)

        if self.add_to_input:
            input = tf.add(input, add_w_input)

        output = self.apply_pool(input, self.kernel_size, pooling_method=pooling_method,
                                 unpooling_method=unpooling_method)
        output = self.zero_center_output(output, center)
        return output

    def apply_pool(self, input, *args, **kwargs):
        raise NotImplementedError()

    def get_description(self):
        return "P{}".format(self.kernel_size)

class Pool2d(Pool):

    def apply_pool(self, input, kernel_size, pooling_method="MAX", *args, **kwargs):
        return pool_2d(input, kernel_size, method=pooling_method, *args, **kwargs)

    def get_description(self):
        return "P_2D{}".format(self.kernel_size)


class UnPool2d(Pool):

    def apply_pool(self, input, kernel_size, unpooling_method="nearest_neighbor", *args, **kwargs):
        return upsample_2d(input, kernel_size, method=unpooling_method, *args, **kwargs)

    def get_description(self):
        return "UP_2D{}".format(self.kernel_size)

# Uses 1x1xC pooling
class Pool3d(Pool):

    def __init__(self, name, kernel_size=1, pooling_method="AVG", add_to_input=False, **kwargs):
        super(Pool3d, self).__init__(**kwargs)
        self.kernel_size = kernel_size
        self.name = name
        self.add_to_input = add_to_input
        self.pooling_method = pooling_method

    def apply_pool(self, input, *args, **kwargs):
        return self.pool_3d(input, *args, **kwargs)

    def pool_3d(self, input, *args, **kwargs):
        if self.pooling_method == "AVG":
            return tf.math.reduce_mean(input, axis=[0,1,2])
        if self.pooling_method == "MAX":
            return tf.math.reduce_max(input, axis=[0,1,2])

    def get_description(self):
        return "P_3D{}".format(self.kernel_size)

