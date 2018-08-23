import tensorflow as tf

from layers.base import Layer
from utilities.activations import lrelu
from utilities.layer_ops import get_incoming_shape


class Conv2d(Layer):
    # global things...
    layer_index = 0

    def __init__(self, kernel_size, output_channels, name,Relu=False,weight_i=None,keep_prob=None, dilation = 1):
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.output_channels = output_channels
        self.name = name
        self.weight= weight_i
        self.keep_prob=keep_prob
        self.Relu=Relu
        
    @staticmethod
    def reverse_global_variables():
        Conv2d.layer_index = 0
 
    def create_layer(self, input):
        self.input_shape = get_incoming_shape(input)
        print('hello',self.input_shape)
        number_of_input_channels = self.input_shape[3]
        self.number_of_input_channels = number_of_input_channels
        with tf.variable_scope('conv', reuse=False):
            if self.weight=='He':
                W = tf.get_variable(('W{}'.format(self.name[-3:])),shape=(self.kernel_size, self.kernel_size,                           number_of_input_channels, self.output_channels), 
                initializer=tf.contrib.layers.variance_scaling_initializer(factor=2.0,mode='FAN_IN',uniform=False),
                regularizer=tf.contrib.layers.l2_regularizer(scale=0.1)
                )
            else:
                W = tf.get_variable(('W{}'.format(self.name[-3:])),shape=(self.kernel_size, self.kernel_size,
                                                            number_of_input_channels,self.output_channels))
            b = tf.Variable(tf.zeros([self.output_channels]))
        #self.encoder_matrix = W
        Conv2d.layer_index += 1

        output = tf.nn.atrous_conv2d(input, W, rate=self.dilation, padding='SAME')
        output = tf.nn.dropout(output, self.keep_prob)
        if self.Relu:
            output=tf.nn.relu(output)
        else:
            output = lrelu(tf.add(tf.contrib.layers.batch_norm(output), b))

        return output

    def create_layer_reversed(self, input, prev_layer=None, reuse=False):
        with tf.variable_scope('conv', reuse=reuse):
            if self.weight=='He':
                W = tf.get_variable('W{}_'.format(self.name[-3:]),shape=(self.kernel_size, self.kernel_size,
                                    self.input_shape[3], self.output_channels),
                    initializer=tf.contrib.layers.variance_scaling_initializer(factor=2.0,mode='FAN_IN',uniform=False ),
                    regularizer=tf.contrib.layers.l2_regularizer(scale=0.1)
                    )
            else:
                W = tf.get_variable('W{}_'.format(self.name[-3:]),
                                shape=(self.kernel_size, self.kernel_size, self.input_shape[3], self.output_channels))
            b = tf.Variable(tf.zeros([W.get_shape().as_list()[2]]))
        output = tf.nn.conv2d_transpose(
            input, W,
            tf.stack([tf.shape(input)[0], self.input_shape[1], self.input_shape[2], self.input_shape[3]]),
            strides=[1,1,1,1], padding='SAME')

        Conv2d.layer_index += 1
        
        output.set_shape([None, self.input_shape[1], self.input_shape[2], self.input_shape[3]])

        output = lrelu(tf.add(tf.contrib.layers.batch_norm(output), b))

        return output


    def get_description(self):
        return "C{},{},{}".format(self.kernel_size, self.output_channels, self.dilation)
