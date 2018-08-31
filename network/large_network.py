import tensorflow as tf

from network.base import Network
from layers.conv2d import Conv2d
from layers.max_pool_2d import MaxPool2d
import tensorflow as tf

class LargeNetwork(Network):

    # actual image dimensions
    IMAGE_HEIGHT = None
    IMAGE_WIDTH = None

    # transformed input dimensions for network input
    FIT_IMAGE_HEIGHT = None
    FIT_IMAGE_WIDTH = None

    IMAGE_CHANNELS = 1

    def __init__(self, weight_init, lr, b1, b2, ep, layer_params, rglzr, act_fn="lrelu", layers=None, **kwargs):
        self.lr = lr
        self.b1 = b1
        self.b2 = b2
        self.ep = ep
        self.rglzr = rglzr

        if layers == None:
            layers = []
            layers.append(Conv2d(kernel_size=layer_params['conv_1_1']['ks'],
                                 dilation=layer_params['conv_1_1']['dilation'], act_fn=act_fn, weight_init=weight_init,
                                 output_channels=['conv_1_1']['output_channels'],
                                 keep_prob=layer_params['conv_1_1']['keep_prob'], name='conv_1_1'))
            layers.append(MaxPool2d(kernel_size=layer_params['max_1']['ks'],
                                    skip_connection=layer_params['max_1']['skip'], name='max_1'))

            layers.append(Conv2d(kernel_size=layer_params['conv_2_1']['ks'],
                                 dilation=layer_params['conv_2_1']['dilation'], act_fn=act_fn, weight_init=weight_init,
                                 output_channels=['conv_2_1']['output_channels'],
                                 keep_prob=layer_params['conv_2_1']['keep_prob'], name='conv_2_1'))

            layers.append(MaxPool2d(kernel_size=layer_params['max_2']['ks'], name='max_2',
                                    skip_connection=layer_params['max_2']['skip']))
            layers.append(Conv2d(kernel_size=layer_params['conv_3_1']['ks'],
                                 dilation=['conv_3_1']['dilation'], act_fn=act_fn, weight_init=weight_init,
                                 output_channels=layer_params['conv_3_1']['output_channels'],
                                 keep_prob=layer_params['conv_3_1']['keep_prob'], name='conv_3_1'))
            layers.append(Conv2d(kernel_size=layer_params['conv_3_2']['ks'],
                                 dilation=layer_params['conv_3_2']['dilation'], act_fn=act_fn, weight_init=weight_init,
                                 output_channels=layer_params['conv_3_2']['output_channels'],
                                 keep_prob=layer_params['conv_3_2']['keep_prob'], name='conv_3_2'))

            layers.append(MaxPool2d(kernel_size=layer_params['max_3']['ks'], name='max_3',
                                    skip_connection=layer_params['max_3']["skip"]))
            layers.append(Conv2d(kernel_size=layer_params['conv_4_1']['ks'],
                                 dilation=['conv_4_1']['dilation'], act_fn=act_fn, weight_init=weight_init,
                                 output_channels=layer_params['conv_4_1']['output_channels'],
                                 keep_prob=layer_params['conv_4_1']['keep_prob'], name='conv_4_1'))
            layers.append(Conv2d(kernel_size=layer_params['conv_4_2']['ks'],
                                 dilation=['conv_4_2']['dilation'], act_fn=act_fn, weight_init=weight_init,
                                 output_channels=layer_params['conv_4_2']['output_channels'],
                                 keep_prob=layer_params['conv_4_2']['keep_prob'], name='conv_4_2'))

            layers.append(MaxPool2d(kernel_size=layer_params['max_4']['ks'], name='max_4',
                                    skip_connection=layer_params['max_4']["skip"]))
            layers.append(Conv2d(kernel_size=layer_params['conv_5_1']['ks'],
                                 dilation=['conv_5_1']['dilation'], act_fn=act_fn, weight_init=weight_init,
                                 output_channels=layer_params['conv_5_1']['output_channels'],
                                 keep_prob=layer_params['conv_5_1']['keep_prob'], name='conv_5_1'))
            layers.append(Conv2d(kernel_size=layer_params['conv_5_2']['ks'],
                                 dilation=['conv_5_2']['dilation'], act_fn=act_fn, weight_init=weight_init,
                                 output_channels=layer_params['conv_5_2']['output_channels'],
                                 keep_prob=layer_params['conv_5_2']['keep_prob'], name='conv_5_2'))

            layers.append(MaxPool2d(kernel_size=layer_params['max_5']['ks'], name='max_5',
                                    skip_connection=layer_params['max_5']["skip"]))
            layers.append(Conv2d(kernel_size=layer_params['conv_6_1']['ks'],
                                 dilation=['conv_6_1']['dilation'], act_fn=act_fn, weight_init=weight_init,
                                 output_channels=layer_params['conv_6_1']['output_channels'],
                                 keep_prob=layer_params['conv_6_1']['keep_prob'], name='conv_6_1'))
            layers.append(Conv2d(kernel_size=layer_params['conv_4_2']['ks'],
                                 dilation=['conv_6_2']['dilation'], act_fn=act_fn, weight_init=weight_init,
                                 output_channels=layer_params['conv_6_2']['output_channels'],
                                 keep_prob=layer_params['conv_6_2']['keep_prob'], name='conv_6_2'))

        self.inputs = tf.placeholder(tf.float32, [None, self.FIT_IMAGE_HEIGHT, self.FIT_IMAGE_WIDTH,
                                                  self.IMAGE_CHANNELS], name='inputs')
        self.targets = tf.placeholder(tf.float32, [None, self.IMAGE_HEIGHT, self.IMAGE_WIDTH, 1], name='targets')
        super(LargeNetwork, self).__init__(layers=layers, **kwargs)
