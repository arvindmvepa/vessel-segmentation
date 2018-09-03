"""This is the file for the base network class"""
import tensorflow as tf
from tensorflow.train.adam import AdamOptimizer
from tensorflow.train.adagrad import AdagradOptimizer
from tensorflow.train.momentum import MomentumOptimizer
from tensorflow.train.adadelta import AdadeltaOptimizer
from tensorflow.train.rmsprop import RMSPropOptimizer

class Network(object):

    def __init__(self, wce_pos_weight=1, regularizer_args=None, learning_rate_and_kwargs=(.001,{}),
                 op_fun_and_kwargs=("adam", {}), layers=None, **kwargs):
        self.cur_learning_rate=learning_rate_and_kwargs
        self.cur_op_fn=op_fun_and_kwargs

        if layers == None :
            raise ValueError("No Layers Defined.")

        self.is_training = tf.placeholder_with_default(False, [], name='is_training')
        self.wce_pos_weight = wce_pos_weight
        self.layer_outputs = []
        self.description = ""
        self.layers = {}
        self.debug1 = self.inputs
        net = self.inputs
        # ENCODER
        for i, layer in enumerate(layers):
            self.layers[layer.name] = net = layer.create_layer(net)
            self.description += "{}".format(layer.get_description())
            self.layer_outputs.append(net)

        print("Number of layers: ", len(layers))
        print("Current input shape: ", net.get_shape())

        layers.reverse()

        # DECODER
        for i, layer in enumerate(layers):
            net=layer.create_layer_reversed(net, prev_layer=self.layers[layer.name])
            self.layer_outputs.append(net)

        self.net_output(net, regularizer_args=regularizer_args, learning_rate_and_kwargs=learning_rate_and_kwargs,
                        op_fun_kwargs=op_fun_kwargs)

    def net_output(self, net, regularizer_args=None, learning_rate_and_kwargs=(.001, {}), op_fun_kwargs=("adam", {})):
        """This method produces the network output"""
        raise NotImplementedError("Not Implemented")

    @property
    def cur_learning_rate(self):
        return self.learning_rate


    @cur_learning_rate.setter
    def cur_learning_rate(self, learning_rate_and_kwargs):
        base_learning_rate, kwargs = learning_rate_and_kwargs
        if kwargs:
            self.global_step = tf.Variable(0, trainable=False)
            self.learning_rate = tf.train.exponential_decay(base_learning_rate, self.global_step, **kwargs)
        else:
            self.learning_rate = base_learning_rate

    @property
    def cur_op_fn(self):
        self.op_fn

    @cur_op_fn.setter
    def cur_op_fn(self, op_fn_and_kwargs):
        op_fn, kwargs = op_fn_and_kwargs
        if op_fn == "adam":
            self.op_fn = AdamOptimizer(learning_rate=self.cur_learning_rate, **kwargs)
        elif op_fn == "adagrad":
            self.op_fn = AdagradOptimizer(learning_rate=self.cur_learning_rate, **kwargs)
        elif op_fn == "momentum":
            self.op_fn = MomentumOptimizer(learning_rate=self.cur_learning_rate, **kwargs)
        elif op_fn == "adadelta":
            self.op_fn = AdadeltaOptimizer(learning_rate=self.cur_learning_rate, **kwargs)
        elif op_fn == "RMSPropOptimizer":
            self.op_fn = RMSPropOptimizer(learning_rate=self.cur_learning_rate, **kwargs)
    @property
    loss

