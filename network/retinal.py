import tensorflow as tf

from network.base import Network
from layers.conv2d import Conv2d
from layers.max_pool_2d import MaxPool2d

class RetinalNetwork(Network):
    IMAGE_HEIGHT = None
    IMAGE_WIDTH = None

    FIT_IMAGE_HEIGHT = None
    FIT_IMAGE_WIDTH = None

    IMAGE_CHANNELS = 1

    def __init__(self, weight_init,learningrate,Beta1,Beta2,epsilon,keep_prob,Layer_param,regularizer=None,Relu=False,layers=None, skip_connections=True,**kwargs):
        # tf.reset_default_graph()
        self.regularizer=regularizer
        self.learningrate=learningrate
        self.Beta1=Beta1
        self.Beta2=Beta2
        self.epsilon=epsilon
        
        if layers == None:

            layers = []
            layers.append(Conv2d(kernel_size=Layer_param['conv_1_1_ks'],dilation=Layer_param['conv_1_1_dilation'], output_channels=Layer_param['conv_1_1_oc'], name='conv_1_1',Relu=Relu,weight_i=weight_init,keep_prob=keep_prob))
            layers.append(MaxPool2d(kernel_size=Layer_param['max_1_ks'], name='max_1', skip_connection=True and skip_connections))

            layers.append(Conv2d(kernel_size=Layer_param['conv_2_1_ks'],dilation=Layer_param['conv_2_1_dilation'], output_channels=Layer_param['conv_2_1_oc'], name='conv_2_1',Relu=Relu,weight_i=weight_init,keep_prob=keep_prob))

            layers.append(MaxPool2d(kernel_size=Layer_param['max_2_ks'], name='max_2', skip_connection=True and skip_connections))
            layers.append(Conv2d(kernel_size=Layer_param['conv_3_1_ks'],dilation=Layer_param['conv_3_1_dilation'], output_channels=Layer_param['conv_3_1_oc'], name='conv_3_1',Relu=Relu, weight_i=weight_init,keep_prob=keep_prob))
            layers.append(Conv2d(kernel_size=Layer_param['conv_3_2_ks'],dilation=Layer_param['conv_3_2_dilation'], output_channels=Layer_param['conv_3_2_oc'], name='conv_3_2',Relu=Relu,weight_i=weight_init,keep_prob=keep_prob))

            layers.append(MaxPool2d(kernel_size=Layer_param['max_3_ks'], name='max_3', skip_connection=True and skip_connections))

            layers.append(Conv2d(kernel_size=Layer_param['conv_4_1_ks'],dilation=Layer_param['conv_4_1_dilation'], output_channels=Layer_param['conv_4_1_oc'], name='conv_4_1',Relu=Relu,weight_i=weight_init,keep_prob=keep_prob))
            layers.append(Conv2d(kernel_size=Layer_param['conv_4_2_ks'],dilation=Layer_param['conv_4_2_dilation'], output_channels=Layer_param['conv_4_2_oc'], name='conv_4_2',Relu=Relu,weight_i=weight_init,keep_prob=keep_prob))

            self.inputs = tf.placeholder(tf.float32, [None, self.FIT_IMAGE_HEIGHT, self.FIT_IMAGE_WIDTH,
                                                      self.IMAGE_CHANNELS], name='inputs')
            self.masks = tf.placeholder(tf.float32, [None, self.IMAGE_HEIGHT, self.IMAGE_WIDTH, 1], name='masks')
            self.targets = tf.placeholder(tf.float32, [None, self.IMAGE_HEIGHT, self.IMAGE_WIDTH, 1], name='targets')
        super(RetinalNetwork, self).__init__(layers=layers, **kwargs)
