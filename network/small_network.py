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

    def __init__(self, weight_init=None, layer_params=None, act_fn="lrelu", act_leak_prob=.2, layers=None, **kwargs):
        self.layer_params = {"conv_1_1":{"ks":3, "dilation":1 , "output_channels":64, "keep_prob":1.0},
                             "max_1": {"ks":2, "skip": True},
                             "conv_2_1": {"ks":3, "dilation":1 , "output_channels":128, "keep_prob":1.0},
                             "max_2": {"ks":2, "skip": True},
                             "conv_3_1": {"ks":3, "dilation":1 , "output_channels":256, "keep_prob":1.0},
                             "conv_3_2": {"ks":3, "dilation":1 , "output_channels":256, "keep_prob":1.0},
                             "max_3": {"ks":2, "skip": True},
                             "conv_4_1": {"ks":7, "dilation":1 , "output_channels":4096, "keep_prob":1.0},
                             "conv_4_2": {"ks":1, "dilation":1 , "output_channels":4096, "keep_prob":1.0}
                             }

        if layer_params:
            self.layer_params.update(layer_params)

        if layers == None:

            layers = list()
            layers.append(Conv2d(kernel_size=self.layer_params['conv_1_1']['ks'],
                                 dilation=self.layer_params['conv_1_1']['dilation'], act_fn=act_fn,
                                 act_leak_prob=act_leak_prob, weight_init=weight_init,
                                 output_channels=self.layer_params['conv_1_1']['output_channels'],
                                 keep_prob=self.layer_params['conv_1_1']['keep_prob'], name='conv_1_1'))
            layers.append(MaxPool2d(kernel_size=self.layer_params['max_1']['ks'],
                                    skip_connection=self.layer_params['max_1']['skip'], name='max_1'))

            layers.append(Conv2d(kernel_size=self.layer_params['conv_2_1']['ks'],
                                 dilation=self.layer_params['conv_2_1']['dilation'], act_fn=act_fn,
                                 act_leak_prob=act_leak_prob, weight_init=weight_init,
                                 output_channels=self.layer_params['conv_2_1']['output_channels'],
                                 keep_prob=self.layer_params['conv_2_1']['keep_prob'], name='conv_2_1'))
            layers.append(MaxPool2d(kernel_size=self.layer_params['max_2']['ks'], name='max_2',
                                    skip_connection=self.layer_params['max_2']['skip']))

            layers.append(Conv2d(kernel_size=self.layer_params['conv_3_1']['ks'],
                                 dilation=self.layer_params['conv_3_1']['dilation'], act_fn=act_fn,
                                 act_leak_prob=act_leak_prob, weight_init=weight_init,
                                 output_channels=self.layer_params['conv_3_1']['output_channels'],
                                 keep_prob=self.layer_params['conv_3_1']['keep_prob'], name='conv_3_1'))
            layers.append(Conv2d(kernel_size=self.layer_params['conv_3_2']['ks'],
                                 dilation=self.layer_params['conv_3_2']['dilation'], act_fn=act_fn,
                                 act_leak_prob=act_leak_prob, weight_init=weight_init,
                                 output_channels=self.layer_params['conv_3_2']['output_channels'],
                                 keep_prob=self.layer_params['conv_3_2']['keep_prob'], name='conv_3_2'))
            layers.append(MaxPool2d(kernel_size=self.layer_params['max_3']['ks'], name='max_3',
                                    skip_connection=self.layer_params['max_3']["skip"]))

            layers.append(Conv2d(kernel_size=self.layer_params['conv_4_1']['ks'],
                                 dilation=self.layer_params['conv_4_1']['dilation'], act_fn=act_fn,
                                 act_leak_prob=act_leak_prob, weight_init=weight_init,
                                 output_channels=self.layer_params['conv_4_1']['output_channels'],
                                 keep_prob=self.layer_params['conv_4_1']['keep_prob'], name='conv_4_1'))
            layers.append(Conv2d(kernel_size=self.layer_params['conv_4_2']['ks'],
                                 dilation=self.layer_params['conv_4_2']['dilation'], act_fn=act_fn,
                                 act_leak_prob=act_leak_prob, weight_init=weight_init,
                                 output_channels=self.layer_params['conv_4_2']['output_channels'],
                                 keep_prob=self.layer_params['conv_4_2']['keep_prob'], name='conv_4_2'))

        super(SmallNetwork, self).__init__(layers=layers, **kwargs)