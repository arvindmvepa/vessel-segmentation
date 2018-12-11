import tensorflow as tf

from layers.base import Layer
from utilities.layer_ops import get_incoming_shape
from utilities.activations import lrelu

class Conv2d(Layer):
    def __init__(self, kernel_size, output_channels, name, batch_norm=True, act_fn="lrelu", act_leak_prob=.2, add_to_input=False,
                 weight_init=None, keep_prob=1.0, dilation=1):
        self.kernel_size = kernel_size
        self.output_channels = output_channels
        self.name = name
        self.batch_norm = batch_norm
        self.act_fn = act_fn
        self.act_leak_prob = act_leak_prob
        self.add_to_input = add_to_input
        self.weight_init = weight_init
        self.keep_prob = keep_prob
        self.dilation = dilation

    def create_layer(self, input, is_training=True, add_w_input=None, center=False, **kwargs):
        if self.add_to_input:
            input = tf.add(input, add_w_input)
        self.input_shape = get_incoming_shape(input)
        print(self.input_shape)
        number_of_input_channels = self.input_shape[3]
        self.number_of_input_channels = number_of_input_channels
        with tf.variable_scope('conv', reuse=False):
            initializer = None
            if self.weight_init == 'He':
                initializer = tf.contrib.layers.variance_scaling_initializer(factor=2.0, mode='FAN_IN', uniform=False)
            elif self.weight_init == 'Xnormal':
                initializer = tf.contrib.layers.xavier_initializer(uniform=False, seed=None)
            W = tf.get_variable(('W{}'.format(self.name[-3:])), shape=(self.kernel_size, self.kernel_size,
                                                                       number_of_input_channels, self.output_channels),
                                initializer=initializer)
            b = tf.Variable(tf.zeros([self.output_channels]))
        tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, W)
        output = tf.nn.atrous_conv2d(input, W, rate=self.dilation, padding='SAME')
        output = tf.nn.dropout(output, self.keep_prob)

        # apply batch-norm
        if self.batch_norm:
            print("apply batch norm")
            print("training {}".format(is_training))
            output = tf.contrib.layers.batch_norm(output, is_training=is_training)
        else:
            print("not applying batch norm")

        output = tf.add(tf.contrib.layers.batch_norm(output), b)

        output = self.get_act_values(output)

        # zero-center activations
        if center:
            print("zero center activation")
            output = output - tf.reduce_mean(output)
        else:
            print("don't zero center activation")
        return output


    def get_act_values(self, input):
        if self.act_fn == "relu":
            return tf.nn.relu(input)
        elif self.act_fn == "lrelu":
           return lrelu(input, self.act_leak_prob)
        elif self.act_fn == "elu":
            return tf.nn.elu(input)
        elif self.act_fn == "maxout":
            return tf.contrib.layers.maxout(input, self.input_shape[3])
        elif self.act_fn is None:
            return input
        else:
            raise ValueError("Activation function {} not recognized".format(self.act_fn))

    def get_description(self):
        return "C{},{},{}".format(self.kernel_size, self.output_channels, self.dilation)


class ConvT2d(Conv2d):
    def create_layer(self, input, is_training=True, add_w_input=None, center=False, **kwargs):
        if self.add_to_input:
            input = tf.add(input, add_w_input)
        self.input_shape = get_incoming_shape(input)
        print(self.input_shape)
        number_of_input_channels = self.input_shape[3]
        self.number_of_input_channels = number_of_input_channels
        with tf.variable_scope('conv', reuse=False):
            initializer = None
            if self.weight_init == 'He':
                initializer = tf.contrib.layers.variance_scaling_initializer(factor=2.0, mode='FAN_IN', uniform=False)
            elif self.weight_init == 'Xnormal':
                initializer = tf.contrib.layers.xavier_initializer(uniform=False, seed=None)
            W = tf.get_variable(('W{}'.format(self.name[-3:])), shape=(self.kernel_size, self.kernel_size,
                                                                       self.output_channels, number_of_input_channels),
                                initializer=initializer)
            b = tf.Variable(tf.zeros([self.output_channels]))
        tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, W)
        # hard-code dimension as `1`, batch size = 1, due to bug
        output = tf.nn.atrous_conv2d_transpose(input, W, tf.stack([1,
                                                                   self.input_shape[1],
                                                                   self.input_shape[2],
                                                                   self.output_channels]),
                                               rate=self.dilation, padding='SAME')
        output = tf.nn.dropout(output, self.keep_prob)
        # apply batch-norm
        if self.batch_norm:
            print("apply batch norm")
            print("training {}".format(is_training))
            output = tf.contrib.layers.batch_norm(output, is_training=is_training)

        output = tf.add(tf.contrib.layers.batch_norm(output), b)

        output = self.get_act_values(output)

        # zero-center activations
        if center:
            print("zero center activation")
            output = output - tf.reduce_mean(output)
        else:
            print("don't zero center activation")
        return output

        return self.get_act_values(output)