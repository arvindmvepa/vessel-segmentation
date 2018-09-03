"""This is the file for the base network class"""
import tensorflow as tf
from tensorflow.python.training.adam import AdamOptimizer
from tensorflow.python.training.adagrad import AdagradOptimizer
from tensorflow.python.training.momentum import MomentumOptimizer
from tensorflow.python.training.adadelta import AdadeltaOptimizer
from tensorflow.python.training.rmsprop import RMSPropOptimizer
from utilities.objective_functions import generalised_dice_loss, sensitivity_specificity_loss, cross_entropy, dice

class Network(object):

    def __init__(self, objective_fn="wce", wce_pos_weight=1, regularizer_args=None, learning_rate_and_kwargs=(.001,{}),
                 op_fun_and_kwargs=("adam", {}), mask=False, layers=None, **kwargs):

        self.cur_objective_fn = objective_fn
        self.cur_learning_rate = learning_rate_and_kwargs
        self.cur_op_fn = op_fun_and_kwargs
        self.regularization = regularizer_args
        self.mask = mask

        self.inputs = tf.placeholder(tf.float32, [None, self.FIT_IMAGE_HEIGHT, self.FIT_IMAGE_WIDTH,
                                                  self.IMAGE_CHANNELS], name='inputs')
        self.targets = tf.placeholder(tf.float32, [None, self.IMAGE_HEIGHT, self.IMAGE_WIDTH, 1], name='targets')
        if self.mask:
            self.masks = tf.placeholder(tf.float32, [None, self.IMAGE_HEIGHT, self.IMAGE_WIDTH, 1], name='masks')

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

        self.calculate_net_output(net)

    def calculate_net_output(self, net):
        net = tf.image.resize_image_with_crop_or_pad(net, self.IMAGE_HEIGHT, self.IMAGE_WIDTH)
        if self.mask:
            net = self.mask_results(net)
        self.segmentation_result = tf.sigmoid(net)
        self.calculate_loss(net)
        self.train_op = self.op_fn.minimize(self.cost, global_step=self._global_step)

    @property
    def cur_learning_rate(self):
        return self.learning_rate


    @cur_learning_rate.setter
    def cur_learning_rate(self, learning_rate_and_kwargs):
        base_learning_rate, kwargs = learning_rate_and_kwargs
        self._global_step = tf.Variable(0, trainable=False)
        if kwargs:
            self.learning_rate = tf.train.exponential_decay(base_learning_rate, self._global_step, **kwargs)
        else:
            self.learning_rate = base_learning_rate

    @property
    def cur_op_fn(self):
        self.op_fn

    @cur_op_fn.setter
    def cur_op_fn(self, op_fn_and_kwargs):
        op_fn, kwargs = op_fn_and_kwargs
        if op_fn == "adam":
            op_cls = AdamOptimizer
        elif op_fn == "adagrad":
            op_cls = AdagradOptimizer
        elif op_fn == "momentum":
            op_cls = MomentumOptimizer
        elif op_fn == "adadelta":
            op_cls = AdadeltaOptimizer
        elif op_fn == "rmsprop":
            op_cls = RMSPropOptimizer
        self.op_fn = op_cls(learning_rate=self.cur_learning_rate, **kwargs)

    @property
    def regularization(self):
        if self.regularizer is not None:
            reg_variables = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            return tf.contrib.layers.apply_regularization(self.regularizer, reg_variables)
        else:
            return 0.0

    @regularization.setter
    def regularization(self, regularizer_args):
        if regularizer_args:
            regularizer_type, regularizer_constant = regularizer_args
            if regularizer_type == "L1":
                self.regularizer = tf.contrib.layers.l1_regularizer(scale=regularizer_constant)
            elif regularizer_type == "L2":
                self.regularizer = tf.contrib.layers.l2_regularizer(scale=regularizer_constant)
            else:
                raise ValueError("Regularizer Type {} Unrecognized".format(regularizer_type))
        else:
            self.regularizer = None

    @property
    def cur_objective_fn(self):
        return self.objective_fn

    @cur_objective_fn.setter
    def cur_objective_fn(self, objective_fn):
        self.objective_fn = self.get_objective_fn(objective_fn)

    # TODO: Consider impact of masking on objective function, seems it would be considered a constant
    # TODO: add options including u-net loss
    def get_objective_fn(self, objective_fn):
        if objective_fn == "ce":
            return lambda targets, net, **kwargs: tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(
                targets, net, pos_weight=1))
        if objective_fn == "wce":
            return lambda targets, net, pos_weight, **kwargs: tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(
                targets, net, pos_weight=pos_weight))
        if objective_fn == "dice":
            return lambda targets, net, **kwargs: dice(net, targets, **kwargs)
        if objective_fn == "gdice":
            return lambda targets, net, **kwargs: generalised_dice_loss(net, targets, **kwargs)
        if objective_fn == "ss":
            return lambda targets, net, **kwargs: sensitivity_specificity_loss(net, targets, **kwargs)

    def mask_results(self, net):
        net = tf.multiply(net, self.masks)
        self.targets = tf.multiply(self.targets, self.masks)
        return net

    def calculate_loss(self, net):
        print('segmentation_result.shape: {}, targets.shape: {}'.format(self.segmentation_result.get_shape(),
                                                                        self.targets.get_shape()))
        self.cost = self.cur_objective_fn(self.targets, net, pos_weight=self.wce_pos_weight) + self.regularization
        self.cost_unweighted = self.get_objective_fn("ce")(self.targets, net) + self.regularization