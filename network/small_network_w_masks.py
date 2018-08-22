import tensorflow as tf

from network.small_network import SmallNetwork
from utilities.mask_ops import mask_op_and_mask_mean

class SmallNetworkWMasks(SmallNetwork):

    def __init__(self, weight_init, lr, b1, b2, ep, layer_params, rglzr, act_fn="lrelu", layers=None, **kwargs):
        self.masks = tf.placeholder(tf.float32, [None, self.IMAGE_HEIGHT, self.IMAGE_WIDTH, 1], name='masks')
        super(SmallNetworkWMasks, self).__init__(weight_init=weight_init, lr=lr, b1=b1, b2=b2, ep=ep,
                                                 layer_params=layer_params, rglzr=rglzr, act_fn=act_fn, layers=layers,
                                                 **kwargs)

    def net_output(self, net):
        net = tf.image.resize_image_with_crop_or_pad(net, self.IMAGE_HEIGHT, self.IMAGE_WIDTH)
        net = tf.multiply(net, self.masks)
        self.segmentation_result = tf.sigmoid(net)
        self.targets = tf.multiply(self.targets, self.masks)
        print('segmentation_result.shape: {}, targets.shape: {}'.format(self.segmentation_result.get_shape(),
                                                                        self.targets.get_shape()))
        # is `reg_term` being optimized?
        reg_term=0
        # get regulairzation terms
        if self.rglzr=='L2':
            regular_variable=tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            reg_term=tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(scale=0.1),regular_variable)
        elif self.regularizer=='L1':
            regular_variable=tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            reg_term=tf.contrib.layers.apply_regularization(tf.contrib.layers.l1_regularizer(scale=0.1),regular_variable)

        self.cost_unweighted = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(self.targets, net, pos_weight=1))
        self.cost = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(self.targets, net,
                                                                            pos_weight=self.wce_pos_weight))
        print('net.shape: {}'.format(net.get_shape()))
        self.train_op = tf.train.AdamOptimizer().minimize(self.cost)
        with tf.name_scope('accuracy'):
            argmax_probs = tf.round(self.segmentation_result)  # 0x1
            correct_pred = tf.cast(tf.equal(argmax_probs, self.targets), tf.float32)
            self.accuracy = mask_op_and_mask_mean(correct_pred, self.masks, 1, self.IMAGE_HEIGHT, self.IMAGE_WIDTH)
            tf.summary.scalar('accuracy', self.accuracy)

        self.summaries = tf.summary.merge_all()