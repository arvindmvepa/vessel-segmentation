"""This is the file for the DSA network subclass"""
from network.base import Network
from layers.conv_ops import Conv2d, ConvT2d
from layers.pool_ops import Pool2d, UnPool2d
from utilities.misc import update

class LargeNetwork(Network):

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
        self.layer_params = {"conv_1_1":{"kernel_size":3, "dilation":1 , "output_channels":64, "batch_norm": batch_norm,
                                         "name": "conv_1_1"},
                             "pool_1": {"kernel_size":2, "name": "pool_1"},
                             "conv_2_1": {"kernel_size":3, "dilation":1 , "output_channels":128,
                                          "batch_norm": batch_norm, "name": "conv_2_1"},
                             "pool_2": {"kernel_size":2, "name": "pool_2"},
                             "conv_3_1": {"kernel_size":3, "dilation":1 , "output_channels":256,
                                          "batch_norm": batch_norm, "name": "conv_3_1"},
                             "conv_3_2": {"kernel_size":3, "dilation":1 , "output_channels":256,
                                          "batch_norm": batch_norm, "name": "conv_3_2"},
                             "pool_3": {"kernel_size":2, "name": "pool_3"},
                             "conv_4_1": {"kernel_size":3, "dilation":1 , "output_channels":512,
                                          "batch_norm": batch_norm, "name": "conv_4_1"},
                             "conv_4_2": {"kernel_size":3, "dilation":1 , "output_channels":512,
                                          "batch_norm": batch_norm, "name": "conv_4_2"},
                             "pool_4": {"kernel_size":2, "name": "pool_4"},
                             "conv_5_1": {"kernel_size":3, "dilation":1 , "output_channels":512,
                                          "batch_norm": batch_norm, "name": "conv_5_1"},
                             "conv_5_2": {"kernel_size":3, "dilation":1 , "output_channels":512,
                                          "batch_norm": batch_norm, "name": "conv_5_2"},
                             "pool_5": {"kernel_size":2, "name": "pool_5"},
                             "conv_6_1": {"kernel_size":7, "dilation":1 , "output_channels":4096,
                                          "batch_norm": batch_norm, "name": "conv_6_1"},
                             "conv_6_2": {"kernel_size":1, "dilation":1 , "output_channels":4096,
                                          "batch_norm": batch_norm, "name": "conv_6_2"},
                             "convt_7_1": {"kernel_size": 1, "dilation": 1, "output_channels": 4096,
                                           "batch_norm": batch_norm, "name": "convt_7_1"},
                             "convt_7_2": {"kernel_size": 7, "dilation": 1, "output_channels": 512,
                                           "batch_norm": batch_norm, "name": "convt_7_2"},
                             "up_8": {"kernel_size": 2, "add_to_input": True, "name": "up_8"},
                             "convt_8_1": {"kernel_size": 3, "dilation": 1, "output_channels": 512,
                                           "batch_norm": batch_norm, "name": "convt_8_1"},
                             "convt_8_2": {"kernel_size": 3, "dilation": 1, "output_channels": 512,
                                           "batch_norm": batch_norm, "name": "convt_8_2"},
                             "up_9": {"kernel_size": 2, "add_to_input": True, "name": "conv_1_1"},
                             "convt_9_1": {"kernel_size": 3, "dilation": 1, "output_channels": 512,
                                           "batch_norm": batch_norm, "name": "convt_9_1"},
                             "convt_9_2": {"kernel_size": 3, "dilation": 1, "output_channels": 256,
                                           "batch_norm": batch_norm, "name": "convt_9_2"},
                             "up_10": {"kernel_size": 2, "add_to_input": True, "name": "up_10"},
                             "convt_10_1": {"kernel_size": 3, "dilation": 1, "output_channels": 256,
                                            "batch_norm": batch_norm, "name": "convt_10_1"},
                             "convt_10_2": {"kernel_size": 3, "dilation": 1, "output_channels": 128,
                                            "batch_norm": batch_norm, "name": "convt_10_2"},
                             "up_11": {"kernel_size": 2, "add_to_input": True, "name": "up_11"},
                             "convt_11_1": {"kernel_size": 3, "dilation": 1, "output_channels": 64,
                                            "batch_norm": batch_norm, "name": "convt_11_1"},
                             "up_12": {"kernel_size": 2, "add_to_input": True, "name": "up_12"},
                             "convt_12_1": {"kernel_size": 3, "dilation": 1, "output_channels": 1,
                                            "batch_norm": batch_norm, "name": "convt_12_1", "act_fn": None},
                             }
        if layer_params:
            self.layer_params = update(self.layer_params,layer_params)

        super(LargeNetwork, self).__init__(weight_init=weight_init, **kwargs)

    def init_encoder(self, **kwargs):
        layers = []
        layers.append(Conv2d(weight_init=self.weight_init, act_fn=self.act_fn, act_leak_prob=self.act_leak_prob,
                             **self.layer_params['conv_1_1']))
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
        layers.append(Pool2d(**self.layer_params['pool_4']))

        layers.append(Conv2d(weight_init=self.weight_init, act_fn=self.act_fn, act_leak_prob=self.act_leak_prob,
                             **self.layer_params['conv_5_1']))
        layers.append(Conv2d(weight_init=self.weight_init, act_fn=self.act_fn, act_leak_prob=self.act_leak_prob,
                             **self.layer_params['conv_5_2']))
        layers.append(Pool2d(**self.layer_params['pool_5']))

        layers.append(Conv2d(weight_init=self.weight_init, act_fn=self.act_fn, act_leak_prob=self.act_leak_prob,
                             **self.layer_params['conv_6_1']))
        layers.append(Conv2d(weight_init=self.weight_init, act_fn=self.act_fn, act_leak_prob=self.act_leak_prob,
                             **self.layer_params['conv_6_2']))
        self.encoder = layers

    def init_decoder(self, **kwargs):
        layers = []
        layers.append(ConvT2d(weight_init=self.weight_init, act_fn=self.act_fn, act_leak_prob=self.act_leak_prob,
                              **self.layer_params['convt_7_1']))
        layers.append(ConvT2d(weight_init=self.weight_init, act_fn=self.act_fn, act_leak_prob=self.act_leak_prob,
                              **self.layer_params['convt_7_2']))

        layers.append(UnPool2d(**self.layer_params['up_8']))
        layers.append(ConvT2d(weight_init=self.weight_init, act_fn=self.act_fn, act_leak_prob=self.act_leak_prob,
                              **self.layer_params['convt_8_1']))
        layers.append(ConvT2d(weight_init=self.weight_init, act_fn=self.act_fn, act_leak_prob=self.act_leak_prob,
                              **self.layer_params['convt_8_2']))

        layers.append(UnPool2d(**self.layer_params['up_9']))
        layers.append(ConvT2d(weight_init=self.weight_init, act_fn=self.act_fn, act_leak_prob=self.act_leak_prob,
                              **self.layer_params['convt_9_1']))
        layers.append(ConvT2d(weight_init=self.weight_init, act_fn=self.act_fn, act_leak_prob=self.act_leak_prob,
                              **self.layer_params['convt_9_2']))

        layers.append(UnPool2d(**self.layer_params['up_10']))
        layers.append(ConvT2d(weight_init=self.weight_init, act_fn=self.act_fn, act_leak_prob=self.act_leak_prob,
                              **self.layer_params['convt_10_1']))
        layers.append(ConvT2d(weight_init=self.weight_init, act_fn=self.act_fn, act_leak_prob=self.act_leak_prob,
                              **self.layer_params['convt_10_2']))

        layers.append(UnPool2d(**self.layer_params['up_11']))
        layers.append(ConvT2d(weight_init=self.weight_init, act_fn=self.act_fn, act_leak_prob=self.act_leak_prob,
                              **self.layer_params['convt_11_1']))

        layers.append(UnPool2d(**self.layer_params['up_12']))
        layers.append(ConvT2d(weight_init=self.weight_init, **self.layer_params['convt_12_1']))
        self.decoder = layers

    def set_layer_op(self, method="AVG", num_prev_last_conv_output_channels=1, *args, **kwargs):
        if num_prev_last_conv_output_channels > 1:
            self.layer_params = update(self.layer_params, {"convt_12_1": {"output_channels":
                                                                             num_prev_last_conv_output_channels}})
        if method == "CONV":
            self.layer_params = update(self.layer_params, {"convt_12_1": {"act_fn": self.act_fn}})
        super(LargeNetwork, self).set_layer_op(method=method, *args, **kwargs)
