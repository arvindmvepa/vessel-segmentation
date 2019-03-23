import tensorflow as tf
import tensorlayer as tl

from layers.base import Layer
from utilities.layer_ops import get_incoming_shape
from utilities.activations import lrelu

class Conv2d(Layer):
    def __init__(self, kernel_size, output_channels, name, batch_norm=True, act_fn="lrelu", act_leak_prob=.2,
                 add_to_input=None, concat_to_input=None, weight_init=None, dp_rate=None, dilation=1, **kwargs):
        super(Conv2d, self).__init__(**kwargs)
        self.kernel_size = kernel_size
        self.output_channels = output_channels
        self.name = name
        self.batch_norm = batch_norm
        self.act_fn = act_fn
        self.act_leak_prob = act_leak_prob
        self.add_to_input = add_to_input
        self.concat_to_input = concat_to_input
        self.weight_init = weight_init
        self.dp_rate = dp_rate
        self.dilation = dilation
        print("name: {}".format(self.name))

    def create_layer(self, input, include_w_input=None, is_training=True, center=False, dp_rate=0.0, **kwargs):
        print("name: {}".format(self.name))
        if self.add_to_input:
            input = tf.add(input, include_w_input)
        if self.concat_to_input:
            input = tf.concat([input, include_w_input],axis=-1)
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
            W = tf.get_variable(('W{}'.format(self.name)), shape=(self.kernel_size, self.kernel_size,
                                                                       number_of_input_channels, self.output_channels),
                                initializer=initializer)
            b = tf.Variable(tf.zeros([self.output_channels]))
        tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, W)
        output = tf.nn.atrous_conv2d(input, W, rate=self.dilation, padding='SAME')
        output = self.apply_dropout(output, dp_rate, is_training)

        # apply batch-norm
        if self.batch_norm:
            print("apply batch norm")
            output = tf.contrib.layers.batch_norm(output, is_training=is_training)

        output = tf.add(tf.contrib.layers.batch_norm(output), b)
        output = self.get_act_values(output)
        output = self.zero_center_output(output, center)
        return output

    def apply_dropout(self, input, dp_rate=0.0, is_training=True):
        if self.dp_rate is not None:
            print("dropout override: {}".format(self.dp_rate))
            return tf.layers.dropout(input, self.dp_rate, training=is_training)
        else:
            return tf.layers.dropout(input, dp_rate, training=is_training)

    def zero_center_output(self, input, center):
        if self.center is not None:
            if self.center:
                print("zero center activation override")
                input = input - tf.reduce_mean(input)
        else:
            if center:
                print("zero center activation normal")
                input = input - tf.reduce_mean(input)
        return input

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
    def create_layer(self, input, include_w_input=None, is_training=True, center=False, dp_rate=0.0, **kwargs):
        print("name: {}".format(self.name))
        if self.add_to_input:
            input = tf.add(input, include_w_input)
        if self.concat_to_input:
            input = tf.concat([input, include_w_input],axis=-1)
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
            W = tf.get_variable(('W{}'.format(self.name)), shape=(self.kernel_size, self.kernel_size,
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
        output = self.apply_dropout(output, dp_rate)
        # apply batch-norm
        if self.batch_norm:
            print("apply batch norm")
            output = tf.contrib.layers.batch_norm(output, is_training=is_training)

        output = tf.add(tf.contrib.layers.batch_norm(output), b)
        output = self.get_act_values(output)
        output = self.zero_center_output(output, center)

        return output


class DeformableConv2d(Layer):
    def __init__(self, prev_layer, offset_layer=None, n_filter=32, filter_size=(3, 3), act_fn=None, name='deformable_conv_2d',
            weight_init=None, bias_init=None, **kwargs):
        super(DeformableConv2d, self).__init__(**kwargs)
        self.prev_layer = prev_layer
        self.offset_layer = offset_layer
        self.n_filter = n_filter
        self.filter_size = filter_size
        self.act_fn = act_fn
        self.name = name
        self.weight_init = weight_init
        self.bias_init = bias_init
        print("name: {}".format(self.name))

    def create_layer(self, input, **kwargs):
        print("name: {}".format(self.name))
        self.input_shape = get_incoming_shape(input)
        print(self.input_shape)
        number_of_input_channels = self.input_shape[3]
        self.number_of_input_channels = number_of_input_channels
        with tf.variable_scope('dfconv', reuse=False):
            if self.weight_init == 'He':
                self.weight_init = tf.contrib.layers.variance_scaling_initializer(factor=2.0, mode='FAN_IN', uniform=False)
            elif self.weight_init == 'Xnormal':
                self.weight_init = tf.contrib.layers.xavier_initializer(uniform=False, seed=None)
        #tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, W)
        output = tl.layers.DeformableConv2d(net, self.prev_layer, self.offset_layer, self.n_filter, self.filter_size, self.act_fn, self.name, self.weight_init)
        output = self.get_act_values(output)
        return output

    def get_act_values(self, input):
        if self.act_fn == "relu":
            return tf.nn.relu(input)
        elif self.act_fn == "lrelu":
           return lrelu(input)
        elif self.act_fn == "elu":
            return tf.nn.elu(input)
        elif self.act_fn == "maxout":
            return tf.contrib.layers.maxout(input, self.input_shape[3])
        elif self.act_fn is None:
            return input
        else:
            raise ValueError("Activation function {} not recognized".format(self.act_fn))

    def get_description(self):
        return "DfConV{},{},{}".format(self.prev_layer, self.n_filter, self.filter_size, self.act_fn)
