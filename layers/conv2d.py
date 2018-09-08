import tensorflow as tf

from layers.base import Layer
from utilities.activations import lrelu
from utilities.layer_ops import get_incoming_shape


class Conv2d(Layer):
    # global things...
    layer_index = 0
    def __init__(self, kernel_size, output_channels, name, act_fn="lrelu", act_leak_prob=.2, weight_init=None,
                 keep_prob=None, dilation = 1):
        self.kernel_size = kernel_size
        self.output_channels = output_channels
        self.name = name
        self.act_fn=act_fn
        self.act_leak_prob = act_leak_prob
        self.weight_init= weight_init
        self.keep_prob=keep_prob
        self.dilation = dilation
        
    @staticmethod
    def reverse_global_variables():
        Conv2d.layer_index = 0
 
    def create_layer(self, input):
        self.input_shape = get_incoming_shape(input)
        print(self.input_shape)
        number_of_input_channels = self.input_shape[3]
        self.number_of_input_channels = number_of_input_channels
        with tf.variable_scope('conv', reuse=False):
            initializer=None
            if self.weight_init == 'He':
                initializer = tf.contrib.layers.variance_scaling_initializer(factor=2.0, mode='FAN_IN', uniform=False)
            elif self.weight_init == 'Xnormal':
                initializer=tf.contrib.layers.xavier_initializer(uniform=False,seed=None)
            W = tf.get_variable(('W{}'.format(self.name[-3:])),shape=(self.kernel_size, self.kernel_size,
                                                                      number_of_input_channels, self.output_channels),
                                initializer=initializer)
            b = tf.Variable(tf.zeros([self.output_channels]))
        tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, W)
        Conv2d.layer_index += 1

        output = tf.nn.atrous_conv2d(input, W, rate=self.dilation, padding='SAME')
        output = tf.nn.dropout(output, self.keep_prob)

        if self.act_fn =="relu":
            output=tf.nn.relu(tf.add(tf.contrib.layers.batch_norm(output), b))
        elif self.act_fn =="lrelu":
            output = lrelu(tf.add(tf.contrib.layers.batch_norm(output), b), self.act_leak_prob)
        else:
            raise ValueError("Activation function {} not recognized".format(self.act_fn))
        return output

    def create_layer_reversed(self, input, prev_layer=None, reuse=False):
        with tf.variable_scope('conv', reuse=reuse):
            initializer=None
            if self.weight_init == 'He':
                initializer = tf.contrib.layers.variance_scaling_initializer(factor=2.0, mode='FAN_IN', uniform=False)
            elif self.weight_init == 'Xnormal':
                initializer=tf.contrib.layers.xavier_initializer(uniform=False,seed=None)
            W = tf.get_variable('W{}_'.format(self.name[-3:]),
                                shape=(self.kernel_size, self.kernel_size, self.input_shape[3], self.output_channels),
                                initializer=initializer)
            b = tf.Variable(tf.zeros([W.get_shape().as_list()[2]]))
        tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, W)
        output = tf.nn.conv2d_transpose(
            input, W,
            tf.stack([tf.shape(input)[0], self.input_shape[1], self.input_shape[2], self.input_shape[3]]),
            strides=[1,1,1,1], padding='SAME')
        output = tf.nn.dropout(output, self.keep_prob)

        Conv2d.layer_index += 1
        output.set_shape([None, self.input_shape[1], self.input_shape[2], self.input_shape[3]])

        if self.act_fn =="relu":
            output = tf.nn.relu(tf.add(tf.contrib.layers.batch_norm(output), b))
        elif self.act_fn =="lrelu":
            output = lrelu(tf.add(tf.contrib.layers.batch_norm(output), b), self.act_leak_prob)
        else:
            raise ValueError("Activation function {} not recognized".format(self.act_fn))
        return output

    def get_description(self):
        return "C{},{},{}".format(self.kernel_size, self.output_channels, self.dilation)
