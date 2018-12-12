from network.base import Network
from layers.conv_ops import Conv2d, ConvT2d
from layers.pool_ops import Pool2d, UnPool2d
from utilities.misc import update

class SmallNetwork(Network):

    # actual image dimensions
    IMAGE_HEIGHT = None
    IMAGE_WIDTH = None

    # transformed input dimensions for network input
    FIT_IMAGE_HEIGHT = None
    FIT_IMAGE_WIDTH = None

    IMAGE_CHANNELS = 1

    def __init__(self, weight_init=None, act_fn="lrelu", act_leak_prob=.2, batch_norm=True, layer_params=None,
                 **kwargs):
        self.weight_init = weight_init
        self.act_fn = act_fn
        self.act_leak_prob = act_leak_prob
        self.layer_params = {"conv_1_1":{"kernel_size":3, "dilation":1 , "output_channels":64, "keep_prob":1.0,
                                         "batch_norm": batch_norm, "name": "conv_1_1"},
                             "pool_1": {"kernel_size":2, "name": "pool_1"},
                             "conv_2_1": {"kernel_size":3, "dilation":1 , "output_channels":128, "keep_prob":1.0,
                                          "batch_norm": batch_norm, "name": "conv_2_1"},
                             "pool_2": {"kernel_size":2, "name": "pool_2"},
                             "conv_3_1": {"kernel_size":3, "dilation":1 , "output_channels":256, "keep_prob":1.0,
                                          "batch_norm": batch_norm, "name": "conv_3_1"},
                             "conv_3_2": {"kernel_size":3, "dilation":1 , "output_channels":256, "keep_prob":1.0,
                                          "batch_norm": batch_norm, "name": "conv_3_2"},
                             "pool_3": {"kernel_size":2, "name": "pool_3"},
                             "conv_4_1": {"kernel_size":7, "dilation":1 , "output_channels":4096, "keep_prob":1.0,
                                          "batch_norm": batch_norm, "name": "conv_4_1"},
                             "conv_4_2": {"kernel_size":1, "dilation":1 , "output_channels":4096, "keep_prob":1.0,
                                          "batch_norm": batch_norm, "name": "conv_4_2"},
                             "convt_5_1": {"kernel_size": 1, "dilation": 1, "output_channels": 4096, "keep_prob": 1.0,
                                           "batch_norm": batch_norm, "name": "convt_5_1"},
                             "convt_5_2": {"kernel_size": 7, "dilation": 1, "output_channels": 256, "keep_prob": 1.0,
                                           "batch_norm": batch_norm, "name": "convt_5_2"},
                             "up_6": {"kernel_size": 2, "add_to_input": True, "name": "up_6"},
                             "convt_6_1": {"kernel_size": 3, "dilation": 1, "output_channels": 256, "keep_prob": 1.0,
                                           "batch_norm": batch_norm, "name": "convt_6_1"},
                             "convt_6_2": {"kernel_size": 3, "dilation": 1, "output_channels": 128, "keep_prob": 1.0,
                                           "batch_norm": batch_norm, "name": "convt_6_2"},
                             "up_7": {"kernel_size": 2, "add_to_input": True, "name": "up_7"},
                             "convt_7_1": {"kernel_size": 3, "dilation": 1, "output_channels": 64, "keep_prob": 1.0,
                                           "batch_norm": batch_norm, "name": "convt_7_1"},
                             "up_8": {"kernel_size": 2, "add_to_input": True, "name": "up_8"},
                             "convt_8_1": {"kernel_size": 3, "dilation": 1, "output_channels": 1, "keep_prob": 1.0,
                                           "batch_norm": batch_norm, "name": "convt_8_1"}}
        if layer_params:
            self.layer_params = update(self.layer_params, layer_params)

        super(SmallNetwork, self).__init__(weight_init=weight_init, **kwargs)


    def init_encoder(self, **kwargs):
        layers = []
        layers.append(Conv2d(weight_init=self.weight_init, act_fn=self.act_fn, act_leak_prob=self.act_leak_prob,
                             **self.layer_params["conv_1_1"]))
        layers.append(Pool2d(**self.layer_params['pool_1']))

        layers.append(Conv2d(weight_init=self.weight_init, act_fn=self.act_fn, act_leak_prob=self.act_leak_prob,
                             **self.layer_params['conv_2_1']))
        layers.append(Pool2d(**self.layer_params['pool_2']))

        layers.append(Conv2d(weight_init=self.weight_init, act_fn=self.act_fn, act_leak_prob=self.act_leak_prob,
                             **self.layer_params['conv_3_1']))
        layers.append(Conv2d(weight_init=self.weight_init, act_fn=self.act_fn, act_leak_prob=self.act_leak_prob,
                             **self.layer_params['conv_3_2']))
        layers.append(Pool2d(**self.layer_params['pool_3']))

        layers.append(Conv2d(weight_init=self.weight_init, act_fn=self.act_fn, act_leak_prob=self.act_leak_prob,
                             **self.layer_params['conv_4_1']))
        layers.append(Conv2d(weight_init=self.weight_init, act_fn=self.act_fn, act_leak_prob=self.act_leak_prob,
                             **self.layer_params['conv_4_2']))
        self.encoder = layers



    def init_decoder(self, **kwargs):
        layers = []
        layers.append(ConvT2d(weight_init=self.weight_init, act_fn=self.act_fn, act_leak_prob=self.act_leak_prob,
                              **self.layer_params['convt_5_1']))
        layers.append(ConvT2d(weight_init=self.weight_init, act_fn=self.act_fn, act_leak_prob=self.act_leak_prob,
                              **self.layer_params['convt_5_2']))

        layers.append(UnPool2d(**self.layer_params['up_6']))
        layers.append(ConvT2d(weight_init=self.weight_init, act_fn=self.act_fn, act_leak_prob=self.act_leak_prob,
                              **self.layer_params['convt_6_1']))
        layers.append(ConvT2d(weight_init=self.weight_init, act_fn=self.act_fn, act_leak_prob=self.act_leak_prob,
                              **self.layer_params['convt_6_2']))

        layers.append(UnPool2d(**self.layer_params['up_7']))
        layers.append(ConvT2d(weight_init=self.weight_init, act_fn=self.act_fn, act_leak_prob=self.act_leak_prob,
                              **self.layer_params['convt_7_1']))

        layers.append(UnPool2d(**self.layer_params['up_8']))
        layers.append(ConvT2d(weight_init=self.weight_init, act_fn=None, **self.layer_params['convt_8_1']))
        self.decoder = layers