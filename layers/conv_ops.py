import tensorflow as tf

from layers.base import Layer
from utilities.layer_ops import get_incoming_shape


class Conv2d(Layer):
    def __init__(self, kernel_size, output_channels, name, act_fn="lrelu", add_to_input=False, weight_init=None,
                 keep_prob=None, dilation = 1):
        self.kernel_size = kernel_size
        self.output_channels = output_channels
        self.name = name
        self.act_fn = act_fn
        self.add_to_input = add_to_input
        self.weight_init = weight_init
        self.keep_prob = keep_prob
        self.dilation = dilation
 
    def create_layer(self, input, add_w_input=None, **kwargs):
        print(self.input_shape)
        if self.add_to_input:
            input = tf.add(input, add_w_input)
        self.input_shape = get_incoming_shape(input)
        number_of_input_channels = self.input_shape[3]
        self.number_of_input_channels = number_of_input_channels
        with tf.variable_scope('conv', reuse=False):
            initializer=None
            if self.weight_init == 'He':
                initializer = tf.contrib.layers.variance_scaling_initializer(factor=2.0, mode='FAN_IN', uniform=False)
            elif self.weight_init == 'Xnormal':
                initializer=tf.contrib.layers.xavier_initializer(uniform=False,seed=None)
            W = tf.get_variable(('W{}'.format(self.name[-3:])), shape=(self.kernel_size, self.kernel_size,
                                                                       number_of_input_channels, self.output_channels),
                                initializer=initializer)
            b = tf.Variable(tf.zeros([self.output_channels]))
        tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, W)
        output = tf.nn.atrous_conv2d(input, W, rate=self.dilation, padding='SAME')
        output = tf.nn.dropout(output, self.keep_prob)
        output = tf.add(tf.contrib.layers.batch_norm(output), b)
        return output

    def get_description(self):
        return "C{},{},{}".format(self.kernel_size, self.output_channels, self.dilation)

class ConvT2d(Conv2d):

    def create_layer(self, input, add_w_input=None, **kwargs):
        print(self.input_shape)
        if self.add_to_input:
            input = tf.add(input, add_w_input)
        self.input_shape = get_incoming_shape(input)
        number_of_input_channels = self.input_shape[3]
        self.number_of_input_channels = number_of_input_channels
        with tf.variable_scope('conv', reuse=False):
            initializer=None
            if self.weight_init == 'He':
                initializer = tf.contrib.layers.variance_scaling_initializer(factor=2.0, mode='FAN_IN', uniform=False)
            elif self.weight_init == 'Xnormal':
                initializer=tf.contrib.layers.xavier_initializer(uniform=False,seed=None)
            W = tf.get_variable(('W{}'.format(self.name[-3:])), shape=(self.kernel_size, self.kernel_size,
                                                                       self.output_channels, number_of_input_channels),
                                initializer=initializer)
            b = tf.Variable(tf.zeros([self.output_channels]))
        tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, W)
        output = tf.nn.conv2d_transpose(input, W, tf.stack([tf.shape(input)[0],
                                                            self.input_shape[1],
                                                            self.input_shape[2],
                                                            self.output_channels]),
                                        strides=[1, 1, 1, 1], padding='SAME')
        output = tf.nn.dropout(output, self.keep_prob)
        output = tf.add(tf.contrib.layers.batch_norm(output), b)
        return output
