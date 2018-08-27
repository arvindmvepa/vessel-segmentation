import tensorflow as tf

from network.base import Network
from layers.conv2d import Conv2d
from layers.max_pool_2d import MaxPool2d

class SmallNetwork(Network):

    # actual image dimensions
    IMAGE_HEIGHT = None
    IMAGE_WIDTH = None

    # transformed input dimensions for network input
    FIT_IMAGE_HEIGHT = None
    FIT_IMAGE_WIDTH = None

    IMAGE_CHANNELS = 1

    def __init__(self, layers=None, skip_connections=True, **kwargs):

        if layers == None:

            layers = []
            layers.append(Conv2d(kernel_size=3, output_channels=64, name='conv_1_1'))
            layers.append(MaxPool2d(kernel_size=2, name='max_1', skip_connection=True and skip_connections))

            layers.append(Conv2d(kernel_size=3, output_channels=128, name='conv_2_1'))

            layers.append(MaxPool2d(kernel_size=2, name='max_2', skip_connection=True and skip_connections))
            layers.append(Conv2d(kernel_size=3, output_channels=256, name='conv_3_1'))
            layers.append(Conv2d(kernel_size=3, dilation=2, output_channels=256, name='conv_3_2'))

            layers.append(MaxPool2d(kernel_size=2, name='max_3', skip_connection=True and skip_connections))

            layers.append(Conv2d(kernel_size=7, output_channels=4096, name='conv_4_1'))
            layers.append(Conv2d(kernel_size=1, output_channels=4096, name='conv_4_2'))

        self.inputs = tf.placeholder(tf.float32, [None, self.FIT_IMAGE_HEIGHT, self.FIT_IMAGE_WIDTH,
                                                  self.IMAGE_CHANNELS], name='inputs')
        self.targets = tf.placeholder(tf.float32, [None, self.IMAGE_HEIGHT, self.IMAGE_WIDTH, 1], name='targets')
        super(SmallNetwork, self).__init__(layers=layers, **kwargs)