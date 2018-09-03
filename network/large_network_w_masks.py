"""This is the file for the DSA network subclass"""
from network.large_network import LargeNetwork
from utilities.mask_ops import mask_op_and_mask_mean
import tensorflow as tf


class LargeNetworkWMasks(LargeNetwork):

    def __init__(self, **kwargs):
        self.masks = tf.placeholder(tf.float32, [None, self.IMAGE_HEIGHT, self.IMAGE_WIDTH, 1], name='masks')
        super(LargeNetworkWMasks, self).__init__(**kwargs)

    def net_output(self, net, regularizer_args=None):
        net = tf.image.resize_image_with_crop_or_pad(net, self.IMAGE_HEIGHT, self.IMAGE_WIDTH)
        net = tf.multiply(net, self.masks)
        self.segmentation_result = tf.sigmoid(net)
        self.targets = tf.multiply(self.targets, self.masks)
        print('segmentation_result.shape: {}, targets.shape: {}'.format(self.segmentation_result.get_shape(),
                                                                        self.targets.get_shape()))

        self.cost_unweighted = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(self.targets, net, pos_weight=1))
        self.cost = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(self.targets, net,
                                                                       pos_weight=self.wce_pos_weight))
        if regularizer_args:
            regularizer_type, regularizer_constant = regularizer_args
            if regularizer_type == "L1":
                regularizer = tf.contrib.layers.l1_regularizer(scale=regularizer_constant)
            elif regularizer_type == "L2":
                regularizer = tf.contrib.layers.l2_regularizer(scale=regularizer_constant)
            else:
                raise ValueError("Regularizer Type {} Unrecognized".format(regularizer_type))
            reg_variables = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            reg_term = tf.contrib.layers.apply_regularization(regularizer, reg_variables)
            self.cost_unweighted = self.cost_unweighted + reg_term
            self.cost = self.cost + reg_term

        print('net.shape: {}'.format(net.get_shape()))
        self.train_op = self.get_op(op_fun, learning_rate=learning_rate, **op_kwargs).minimize(self.cost,
                                                                                               global_step=self.global_step)
        with tf.name_scope('accuracy'):
            argmax_probs = tf.round(self.segmentation_result)  # 0x1
            correct_pred = tf.cast(tf.equal(argmax_probs, self.targets), tf.float32)
            self.accuracy = mask_op_and_mask_mean(correct_pred, self.masks, 1, self.IMAGE_HEIGHT, self.IMAGE_WIDTH)
            tf.summary.scalar('accuracy', self.accuracy)

        self.summaries = tf.summary.merge_all()