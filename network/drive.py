"""This is the file for the DRIVE network subclass"""
import tensorflow as tf

from network.base import Network
from layers.conv2d import Conv2d
from layers.max_pool_2d import MaxPool2d
from utilities.mask_ops import mask_op_and_mask_mean

class DriveNetwork(Network):
    # actual image dimensions
    IMAGE_HEIGHT = 584
    IMAGE_WIDTH = 565

    # transformed input dimensions for network input
    FIT_IMAGE_HEIGHT = 584
    FIT_IMAGE_WIDTH = 584

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
            # tensors for masks are also specified
            self.masks = tf.placeholder(tf.float32, [None, self.IMAGE_HEIGHT, self.IMAGE_WIDTH, 1], name='masks')
            self.targets = tf.placeholder(tf.float32, [None, self.IMAGE_HEIGHT, self.IMAGE_WIDTH, 1], name='targets')
        super(DriveNetwork, self).__init__(layers=layers, **kwargs)

    def net_output(self, net):
        # the images dimensions are reduced given the actual input image heigh and width
        net = tf.image.resize_image_with_crop_or_pad(net, self.IMAGE_HEIGHT, self.IMAGE_WIDTH)
        # only the result within the mask are included
        net = tf.multiply(net, self.masks)
        # produce network predictions
        self.segmentation_result = tf.sigmoid(net)
        # only the targets within the mask are included
        self.targets = tf.multiply(self.targets, self.masks)
        print('segmentation_result.shape: {}, targets.shape: {}'.format(self.segmentation_result.get_shape(),
                                                                        self.targets.get_shape()))
        # get unweighted loss
        self.cost_unweighted = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(self.targets, net, pos_weight=1))
        # get weighted loss
        self.cost = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(self.targets, net,
                                                                            pos_weight=self.wce_pos_weight))
        print('net.shape: {}'.format(net.get_shape()))
        # optimize loss
        self.train_op = tf.train.AdamOptimizer().minimize(self.cost)
        # calculate accuracy
        with tf.name_scope('accuracy'):
            argmax_probs = tf.round(self.segmentation_result)  # 0x1
            correct_pred = tf.cast(tf.equal(argmax_probs, self.targets), tf.float32)
            self.accuracy = mask_op_and_mask_mean(correct_pred, self.masks, 1, self.IMAGE_HEIGHT, self.IMAGE_WIDTH)
            tf.summary.scalar('accuracy', self.accuracy)

        self.summaries = tf.summary.merge_all()