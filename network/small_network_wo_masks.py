import tensorflow as tf

from network.small_network import SmallNetwork

class SmallNetworkWoMasks(SmallNetwork):

    def __init__(self, **kwargs):
        super(SmallNetworkWoMasks, self).__init__(**kwargs)

    def net_output(self, net, regularizer_args=None, learning_rate_and_kwargs=(.001, {}), op_fun_kwargs=("adam", {})):
        learning_rate, learning_rate_kwargs = learning_rate_and_kwargs
        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(learning_rate, global_step, **learning_rate_kwargs)

        op_fun, op_kwargs = op_fun_kwargs
        net = tf.image.resize_image_with_crop_or_pad(net, self.IMAGE_HEIGHT, self.IMAGE_WIDTH)
        self.segmentation_result = tf.sigmoid(net)
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
                                                                                               global_step=global_step)
        with tf.name_scope('accuracy'):
            argmax_probs = tf.round(self.segmentation_result)  # 0x1
            correct_pred = tf.cast(tf.equal(argmax_probs, self.targets), tf.float32)
            self.accuracy = tf.reduce_mean(correct_pred)
            tf.summary.scalar('accuracy', self.accuracy)
        self.summaries = tf.summary.merge_all()