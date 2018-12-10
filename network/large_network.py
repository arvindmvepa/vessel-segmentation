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
        self.layer_params = {"conv_1_1":{"ks":3, "dilation":1 , "output_channels":64, "keep_prob":1.0,
                                         "batch_norm": batch_norm},
                             "pool_1": {"ks":2},
                             "conv_2_1": {"ks":3, "dilation":1 , "output_channels":128, "keep_prob":1.0,
                                          "batch_norm": batch_norm},
                             "pool_2": {"ks":2},
                             "conv_3_1": {"ks":3, "dilation":1 , "output_channels":256, "keep_prob":1.0,
                                          "batch_norm": batch_norm},
                             "conv_3_2": {"ks":3, "dilation":1 , "output_channels":256, "keep_prob":1.0,
                                          "batch_norm": batch_norm},
                             "pool_3": {"ks":2},
                             "conv_4_1": {"ks":3, "dilation":1 , "output_channels":512, "keep_prob":1.0,
                                          "batch_norm": batch_norm},
                             "conv_4_2": {"ks":3, "dilation":1 , "output_channels":512, "keep_prob":1.0,
                                          "batch_norm": batch_norm},
                             "pool_4": {"ks":2},
                             "conv_5_1": {"ks":3, "dilation":1 , "output_channels":512, "keep_prob":1.0,
                                          "batch_norm": batch_norm},
                             "conv_5_2": {"ks":3, "dilation":1 , "output_channels":512, "keep_prob":1.0,
                                          "batch_norm": batch_norm},
                             "pool_5": {"ks":2},
                             "conv_6_1": {"ks":7, "dilation":1 , "output_channels":4096, "keep_prob":1.0,
                                          "batch_norm": batch_norm},
                             "conv_6_2": {"ks":1, "dilation":1 , "output_channels":4096, "keep_prob":1.0,
                                          "batch_norm": batch_norm},

                             "convt_7_1": {"ks": 1, "dilation": 1, "output_channels": 4096, "keep_prob": 1.0,
                                           "batch_norm": batch_norm},
                             "convt_7_2": {"ks": 7, "dilation": 1, "output_channels": 512, "keep_prob": 1.0,
                                           "batch_norm": batch_norm},
                             "up_8": {"ks": 2, "add_to_input": True},
                             "convt_8_1": {"ks": 3, "dilation": 1, "output_channels": 512, "keep_prob": 1.0,
                                           "batch_norm": batch_norm},
                             "convt_8_2": {"ks": 3, "dilation": 1, "output_channels": 512, "keep_prob": 1.0,
                                           "batch_norm": batch_norm},
                             "up_9": {"ks": 2, "add_to_input": True},
                             "convt_9_1": {"ks": 3, "dilation": 1, "output_channels": 512, "keep_prob": 1.0,
                                           "batch_norm": batch_norm},
                             "convt_9_2": {"ks": 3, "dilation": 1, "output_channels": 256, "keep_prob": 1.0,
                                           "batch_norm": batch_norm},
                             "up_10": {"ks": 2, "add_to_input": True},
                             "convt_10_1": {"ks": 3, "dilation": 1, "output_channels": 256, "keep_prob": 1.0,
                                            "batch_norm": batch_norm},
                             "convt_10_2": {"ks": 3, "dilation": 1, "output_channels": 128, "keep_prob": 1.0,
                                            "batch_norm": batch_norm},
                             "up_11": {"ks": 2, "add_to_input": True},
                             "convt_11_1": {"ks": 3, "dilation": 1, "output_channels": 64, "keep_prob": 1.0,
                                            "batch_norm": batch_norm},
                             "up_12": {"ks": 2, "add_to_input": True},
                             "convt_12_1": {"ks": 3, "dilation": 1, "output_channels": 1, "keep_prob": 1.0,
                                            "batch_norm": batch_norm},
                             }
        if layer_params:
            update(self.layer_params.update,layer_params)

        super(LargeNetwork, self).__init__(weight_init=weight_init, **kwargs)

    def init_encoder(self, **kwargs):
        layers = []
        layers.append(Conv2d(kernel_size=self.layer_params['conv_1_1']['ks'],
                             dilation=self.layer_params['conv_1_1']['dilation'],
                             weight_init=self.weight_init, act_fn=self.act_fn, act_leak_prob=self.act_leak_prob,
                             output_channels=self.layer_params['conv_1_1']['output_channels'],
                             keep_prob=self.layer_params['conv_1_1']['keep_prob'],
                             batch_norm=self.layer_params['conv_1_1']['batch_norm'], name='conv_1_1'))
        layers.append(Pool2d(kernel_size=self.layer_params['pool_1']['ks'], name='pool_1'))

        layers.append(Conv2d(kernel_size=self.layer_params['conv_2_1']['ks'],
                             dilation=self.layer_params['conv_2_1']['dilation'],
                             weight_init=self.weight_init, act_fn=self.act_fn, act_leak_prob=self.act_leak_prob,
                             output_channels=self.layer_params['conv_2_1']['output_channels'],
                             keep_prob=self.layer_params['conv_2_1']['keep_prob'],
                             batch_norm=self.layer_params['conv_2_1']['batch_norm'], name='conv_2_1'))
        layers.append(Pool2d(kernel_size=self.layer_params['pool_2']['ks'], name='pool_2'))

        layers.append(Conv2d(kernel_size=self.layer_params['conv_3_1']['ks'],
                             dilation=self.layer_params['conv_3_1']['dilation'],
                             weight_init=self.weight_init, act_fn=self.act_fn, act_leak_prob=self.act_leak_prob,
                             output_channels=self.layer_params['conv_3_1']['output_channels'],
                             keep_prob=self.layer_params['conv_3_1']['keep_prob'],
                             batch_norm=self.layer_params['conv_3_1']['batch_norm'], name='conv_3_1'))
        layers.append(Conv2d(kernel_size=self.layer_params['conv_3_2']['ks'],
                             dilation=self.layer_params['conv_3_2']['dilation'], act_fn=self.act_fn,
                             weight_init=self.weight_init,
                             output_channels=self.layer_params['conv_3_2']['output_channels'],
                             keep_prob=self.layer_params['conv_3_2']['keep_prob'],
                             batch_norm=self.layer_params['conv_3_2']['batch_norm'], name='conv_3_2'))
        layers.append(Pool2d(kernel_size=self.layer_params['pool_3']['ks'], name='pool_3'))

        layers.append(Conv2d(kernel_size=self.layer_params['conv_4_1']['ks'],
                             dilation=self.layer_params['conv_4_1']['dilation'],
                             weight_init=self.weight_init, act_fn=self.act_fn, act_leak_prob=self.act_leak_prob,
                             output_channels=self.layer_params['conv_4_1']['output_channels'],
                             keep_prob=self.layer_params['conv_4_1']['keep_prob'],
                             batch_norm=self.layer_params['conv_4_1']['batch_norm'], name='conv_4_1'))
        layers.append(Conv2d(kernel_size=self.layer_params['conv_4_2']['ks'],
                             dilation=self.layer_params['conv_4_2']['dilation'],
                             weight_init=self.weight_init, act_fn=self.act_fn, act_leak_prob=self.act_leak_prob,
                             output_channels=self.layer_params['conv_4_2']['output_channels'],
                             keep_prob=self.layer_params['conv_4_2']['keep_prob'],
                             batch_norm=self.layer_params['conv_4_2']['batch_norm'], name='conv_4_2'))
        layers.append(Pool2d(kernel_size=self.layer_params['pool_4']['ks'], name='pool_4'))

        layers.append(Conv2d(kernel_size=self.layer_params['conv_5_1']['ks'],
                             dilation=self.layer_params['conv_5_1']['dilation'],
                             weight_init=self.weight_init, act_fn=self.act_fn, act_leak_prob=self.act_leak_prob,
                             output_channels=self.layer_params['conv_5_1']['output_channels'],
                             keep_prob=self.layer_params['conv_5_1']['keep_prob'],
                             batch_norm=self.layer_params['conv_5_1']['batch_norm'], name='conv_5_1'))
        layers.append(Conv2d(kernel_size=self.layer_params['conv_5_2']['ks'],
                             dilation=self.layer_params['conv_5_2']['dilation'],
                             weight_init=self.weight_init, act_fn=self.act_fn, act_leak_prob=self.act_leak_prob,
                             output_channels=self.layer_params['conv_5_2']['output_channels'],
                             keep_prob=self.layer_params['conv_5_2']['keep_prob'],
                             batch_norm=self.layer_params['conv_5_2']['batch_norm'], name='conv_5_2'))
        layers.append(Pool2d(kernel_size=self.layer_params['pool_5']['ks'], name='pool_5'))

        layers.append(Conv2d(kernel_size=self.layer_params['conv_6_1']['ks'],
                             dilation=self.layer_params['conv_6_1']['dilation'],
                             weight_init=self.weight_init, act_fn=self.act_fn, act_leak_prob=self.act_leak_prob,
                             output_channels=self.layer_params['conv_6_1']['output_channels'],
                             keep_prob=self.layer_params['conv_6_1']['keep_prob'],
                             batch_norm=self.layer_params['conv_6_1']['batch_norm'], name='conv_6_1'))
        layers.append(Conv2d(kernel_size=self.layer_params['conv_6_2']['ks'],
                             dilation=self.layer_params['conv_6_2']['dilation'],
                             weight_init=self.weight_init, act_fn=self.act_fn, act_leak_prob=self.act_leak_prob,
                             output_channels=self.layer_params['conv_6_2']['output_channels'],
                             keep_prob=self.layer_params['conv_6_2']['keep_prob'],
                             batch_norm=self.layer_params['conv_6_2']['batch_norm'], name='conv_6_2'))
        self.encoder = layers

    def init_decoder(self, **kwargs):
        layers = []
        layers.append(ConvT2d(kernel_size=self.layer_params['convt_7_1']['ks'],
                              dilation=self.layer_params['convt_7_1']['dilation'],
                              weight_init=self.weight_init, act_fn=self.act_fn, act_leak_prob=self.act_leak_prob,
                              output_channels=self.layer_params['convt_7_1']['output_channels'],
                              keep_prob=self.layer_params['convt_7_1']['keep_prob'],
                              batch_norm=self.layer_params['convt_7_1']['batch_norm'], name='convt_7_1'))
        layers.append(ConvT2d(kernel_size=self.layer_params['convt_7_2']['ks'],
                              dilation=self.layer_params['convt_7_2']['dilation'],
                              weight_init=self.weight_init, act_fn=self.act_fn, act_leak_prob=self.act_leak_prob,
                              output_channels=self.layer_params['convt_7_2']['output_channels'],
                              keep_prob=self.layer_params['convt_7_2']['keep_prob'],
                              batch_norm=self.layer_params['convt_7_2']['batch_norm'], name='convt_7_2'))

        layers.append(UnPool2d(kernel_size=self.layer_params['up_8']['ks'], name='up_8',
                               add_to_input=self.layer_params['up_8']["add_to_input"]))
        layers.append(ConvT2d(kernel_size=self.layer_params['convt_8_1']['ks'],
                              dilation=self.layer_params['convt_8_1']['dilation'],
                              weight_init=self.weight_init, act_fn=self.act_fn, act_leak_prob=self.act_leak_prob,
                              output_channels=self.layer_params['convt_8_1']['output_channels'],
                              keep_prob=self.layer_params['convt_8_1']['keep_prob'],
                              batch_norm=self.layer_params['convt_8_1']['batch_norm'], name='convt_8_1'))
        layers.append(ConvT2d(kernel_size=self.layer_params['convt_8_2']['ks'],
                              dilation=self.layer_params['convt_8_2']['dilation'],
                              weight_init=self.weight_init, act_fn=self.act_fn, act_leak_prob=self.act_leak_prob,
                              output_channels=self.layer_params['convt_8_2']['output_channels'],
                              keep_prob=self.layer_params['convt_8_2']['keep_prob'],
                              batch_norm=self.layer_params['convt_8_2']['batch_norm'], name='convt_8_2'))

        layers.append(UnPool2d(kernel_size=self.layer_params['up_9']['ks'], name='up_9',
                               add_to_input=self.layer_params['up_9']["add_to_input"]))
        layers.append(ConvT2d(kernel_size=self.layer_params['convt_9_1']['ks'],
                              dilation=self.layer_params['convt_9_1']['dilation'],
                              weight_init=self.weight_init, act_fn=self.act_fn, act_leak_prob=self.act_leak_prob,
                              output_channels=self.layer_params['convt_9_1']['output_channels'],
                              keep_prob=self.layer_params['convt_9_1']['keep_prob'],
                              batch_norm=self.layer_params['convt_9_1']['batch_norm'], name='convt_9_1'))
        layers.append(ConvT2d(kernel_size=self.layer_params['convt_9_2']['ks'],
                              dilation=self.layer_params['convt_9_2']['dilation'],
                              weight_init=self.weight_init, act_fn=self.act_fn, act_leak_prob=self.act_leak_prob,
                              output_channels=self.layer_params['convt_9_2']['output_channels'],
                              keep_prob=self.layer_params['convt_9_2']['keep_prob'],
                              batch_norm=self.layer_params['convt_9_2']['batch_norm'], name='convt_9_2'))

        layers.append(UnPool2d(kernel_size=self.layer_params['up_10']['ks'], name='up_10',
                               add_to_input=self.layer_params['up_10']["add_to_input"]))
        layers.append(ConvT2d(kernel_size=self.layer_params['convt_10_1']['ks'],
                              dilation=self.layer_params['convt_10_1']['dilation'],
                              weight_init=self.weight_init, act_fn=self.act_fn, act_leak_prob=self.act_leak_prob,
                              output_channels=self.layer_params['convt_10_1']['output_channels'],
                              keep_prob=self.layer_params['convt_10_1']['keep_prob'],
                              batch_norm=self.layer_params['convt_10_1']['batch_norm'], name='convt_10_1'))
        layers.append(ConvT2d(kernel_size=self.layer_params['convt_10_2']['ks'],
                              dilation=self.layer_params['convt_10_2']['dilation'],
                              weight_init=self.weight_init, act_fn=self.act_fn, act_leak_prob=self.act_leak_prob,
                              output_channels=self.layer_params['convt_10_2']['output_channels'],
                              keep_prob=self.layer_params['convt_10_2']['keep_prob'],
                              batch_norm=self.layer_params['convt_10_2']['batch_norm'], name='convt_10_2'))

        layers.append(UnPool2d(kernel_size=self.layer_params['up_11']['ks'], name='up_11',
                               add_to_input=self.layer_params['up_11']["add_to_input"]))
        layers.append(ConvT2d(kernel_size=self.layer_params['convt_11_1']['ks'],
                              dilation=self.layer_params['convt_11_1']['dilation'],
                              weight_init=self.weight_init, act_fn=self.act_fn, act_leak_prob=self.act_leak_prob,
                              output_channels=self.layer_params['convt_11_1']['output_channels'],
                              keep_prob=self.layer_params['convt_11_1']['keep_prob'],
                              batch_norm=self.layer_params['convt_11_1']['batch_norm'], name='convt_11_1'))

        layers.append(UnPool2d(kernel_size=self.layer_params['up_12']['ks'], name='up_12',
                               add_to_input=self.layer_params['up_12']["add_to_input"]))
        layers.append(ConvT2d(kernel_size=self.layer_params['convt_12_1']['ks'],
                              dilation=self.layer_params['convt_12_1']['dilation'],
                              weight_init=self.weight_init, act_fn=None,
                              output_channels=self.layer_params['convt_12_1']['output_channels'],
                              keep_prob=self.layer_params['convt_12_1']['keep_prob'],
                              batch_norm=self.layer_params['convt_12_1']['batch_norm'], name='convt_12_1'))
        self.decoder = layers