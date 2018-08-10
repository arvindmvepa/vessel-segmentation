import tensorflow as tf

from network.base import Network
from layers.conv2d import Conv2d
from layers.max_pool_2d import MaxPool2d
from utilities.mask_ops import mask_op_and_mask_mean

class DriveNetwork(Network):
    IMAGE_WIDTH = 584
    IMAGE_HEIGHT = 565

    FIT_IMAGE_WIDTH = 584
    FIT_IMAGE_HEIGHT = 584

    IMAGE_CHANNELS = 1

    def __init__(self, layers=None, skip_connections=True, **kwargs):
        super(DriveNetwork, self).__init__(**kwargs)

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

            self.inputs = tf.placeholder(tf.float32,
                                         [None, self.IMAGE_HEIGHT, self.IMAGE_WIDTH, self.IMAGE_CHANNELS],
                                         name='inputs')

            self.masks = tf.placeholder(tf.float32, [None, self.IMAGE_WIDTH, self.IMAGE_HEIGHT, 1], name='masks')
            super(DriveNetwork, self).__init__(layers=layers, IMAGE_WIDTH=self.IMAGE_WIDTH, IMAGE_HEIGHT=self.IMAGE_HEIGHT,
                                               FIT_IMAGE_WIDTH=self.FIT_IMAGE_WIDTH,
                                               FIT_IMAGE_HEIGHT=self.FIT_IMAGE_HEIGHT, **kwargs)

        self.inputs = tf.placeholder(tf.float32, [None, DriveNetwork.FIT_IMAGE_WIDTH, DriveNetwork.FIT_IMAGE_HEIGHT, DriveNetwork.IMAGE_CHANNELS],
                                     name='inputs')
        self.targets = tf.placeholder(tf.float32, [None, DriveNetwork.IMAGE_WIDTH, DriveNetwork.IMAGE_HEIGHT, 1], name='targets')
        self.is_training = tf.placeholder_with_default(True, [], name='is_training')
        self.layer_outputs = []
        self.description = ""
        self.layers = {}
        self.debug1 = self.inputs
        net = self.inputs

        # ENCODER
        for i, layer in enumerate(layers):
            self.layers[layer.name] = net = layer.create_layer(net)
            self.description += "{}".format(layer.get_description())
            self.layer_outputs.append(net)

        print("Number of layers: ", len(layers))
        print("Current input shape: ", net.get_shape())

        layers.reverse()

        # DECODER
        for i, layer in enumerate(layers):
            net = layer.create_layer_reversed(net, prev_layer=self.layers[layer.name])
            self.layer_outputs.append(net)

        net = tf.image.resize_image_with_crop_or_pad(net, DriveNetwork.IMAGE_WIDTH, DriveNetwork.IMAGE_HEIGHT)

        net = tf.multiply(net, self.masks)
        self.segmentation_result = tf.sigmoid(net)
        self.targets = tf.multiply(self.targets, self.masks)
        print('segmentation_result.shape: {}, targets.shape: {}'.format(self.segmentation_result.get_shape(),
                                                                        self.targets.get_shape()))

        self.cost_unweighted = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(self.targets, net, pos_weight=1))
        self.cost = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(self.targets, net,
                                                                            pos_weight=self.wce_pos_weight))
        print('net.shape: {}'.format(net.get_shape()))
        self.train_op = tf.train.AdamOptimizer().minimize(self.cost)
        with tf.name_scope('accuracy'):
            argmax_probs = tf.round(self.segmentation_result)  # 0x1
            correct_pred = tf.cast(tf.equal(argmax_probs, self.targets), tf.float32)
            self.accuracy = mask_op_and_mask_mean(correct_pred, self.masks, 1, self.IMAGE_WIDTH, self.IMAGE_HEIGHT)
            tf.summary.scalar('accuracy', self.accuracy)

        self.summaries = tf.summary.merge_all()