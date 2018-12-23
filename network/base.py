"""This is the file for the base network class"""
import tensorflow as tf
from tensorflow.python.training.adam import AdamOptimizer
from tensorflow.python.training.adagrad import AdagradOptimizer
from tensorflow.python.training.momentum import MomentumOptimizer
from tensorflow.python.training.adadelta import AdadeltaOptimizer
from tensorflow.python.training.rmsprop import RMSPropOptimizer
from tensorflow.python.training.gradient_descent import GradientDescentOptimizer

from utilities.objective_functions import generalised_dice_loss, sensitivity_specificity_loss, cross_entropy, dice
from layers.conv_ops import Conv2d
from layers.pool_ops import Pool3d

class Network(object):

    def __init__(self, objective_fn="wce", weight_init = None, regularizer_args=None,
                 learning_rate_and_kwargs=(.001, {}), op_fun_and_kwargs=("adam", {}), mask=False, dp_rate=0.0,
                 center=False, pooling_method="MAX", unpooling_method="nearest_neighbor", last_layer_op=None,
                 num_prev_last_conv_output_channels=1, layers=None, encoder_decoder=True, num_batches_in_epoch = 1,
                 **kwargs):
        self.num_batches_in_epoch = num_batches_in_epoch
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

        self.is_training = tf.placeholder_with_default(False, [], name='is_training')
        self.layer_outputs = []
        self.description = ""
        self.layers = {}
        self.debug1 = self.inputs

        print("networking training: {}".format(self.is_training))

        if last_layer_op:
            print("Set Last Layer Op: {}".format(last_layer_op))
            self.set_layer_op(weight_init=weight_init, method=last_layer_op,
                              num_prev_last_conv_output_channels=num_prev_last_conv_output_channels)

        if encoder_decoder:
            self.init_encoder(**kwargs)
            self.init_decoder(**kwargs)

            print("Number of Encoder Layers: ", len(self.encoder))
            print("Number of Decoder Layers: ", len(self.decoder))

            net = self.encode(self.inputs, center=center, pooling_method=pooling_method, dp_rate=dp_rate)
            net = self.decode(net, center=center, unpooling_method=unpooling_method, dp_rate=dp_rate)
        else:
            if layers == None:
                raise ValueError("No Layers Defined.")
            print("Number of layers: ", len(layers))
            for i, layer in enumerate(layers):
                self.layers[i] = net = layer.create_layer(net, is_training=self.is_training, center=center,
                                                          pooling_method=pooling_method,
                                                          unpooling_method=unpooling_method)
                self.description += "{}".format(layer.get_description())
                self.layer_outputs.append(net)

        print("Current output shape: ", net.get_shape())

        if last_layer_op:
            net = self.apply_last_layer_op(net, is_training=self.is_training)
            print("Additional Last Layer Applied: {}".format(last_layer_op))

        self.calculate_net_output(net, **kwargs)

    def calculate_net_output(self, net,  **loss_kwargs):
        net = tf.image.resize_image_with_crop_or_pad(net, self.IMAGE_HEIGHT, self.IMAGE_WIDTH)
        if self.mask:
            net = self.mask_results(net)
        self.segmentation_result = tf.sigmoid(net)
        self.calculate_loss(net, **loss_kwargs)
        self.train_op = self.cur_op_fn.minimize(self.cost, global_step=self._global_step)

    def init_encoder(self, **encoder_kwargs):
        raise NotImplementedError("Method Not Implemented")

    def init_decoder(self, **decoder_kwargs):
        raise NotImplementedError("Method Not Implemented")

    def encode(self, net, center=False, pooling_method="MAX", dp_rate=0.0):
        for i, layer in enumerate(self.encoder):
            self.layers[i] = net = layer.create_layer(net, is_training=self.is_training, center=center,
                                                      pooling_method=pooling_method, dp_rate=dp_rate)
            self.description += "{}".format(layer.get_description())
            self.layer_outputs.append(net)
        return net


    def decode(self, net, center=False, unpooling_method="MAX", dp_rate=0.0):
        for i, layer in enumerate(self.decoder, start=1):
            net = layer.create_layer(net, add_w_input=self.layers[len(self.decoder) - i], is_training=self.is_training,
                                     center=center, unpooling_method=unpooling_method, dp_rate=dp_rate)
            self.description += "{}".format(layer.get_description())
            self.layer_outputs.append(net)
        return net

    def calculate_loss(self, net, **kwargs):
        print('segmentation_result.shape: {}, targets.shape: {}'.format(self.segmentation_result.get_shape(),
                                                                        self.targets.get_shape()))
        self.cost = self.cur_objective_fn(self.targets, net, **kwargs) + self.regularization
        self.cost_unweighted = self.get_objective_fn("ce")(self.targets, net) + self.regularization

    def apply_last_layer_op(self, net, **kwargs):
        print("apply last layer op")
        return self.last_layer_op.create_layer(net, **kwargs)

    def mask_results(self, net):
        net = tf.multiply(net, self.masks)
        self.targets = tf.multiply(self.targets, self.masks)
        return net

    def set_layer_op(self, weight_init=None, method="AVG", *args, **kwargs):
        if method == "AVG" or method == "MAX":
            self.last_layer_op = Pool3d(pooling_method=method, name='last_pool')
        elif method == "CONV":
            self.last_layer_op = Conv2d(kernel_size=1, dilation=1,  weight_init=weight_init, act_fn=None,
                                        output_channels=1, name='last_conv')
        else:
            raise ValueError("No last layer op for method: {}".format(method))

    @property
    def cur_learning_rate(self):
        return self.learning_rate


    @cur_learning_rate.setter
    def cur_learning_rate(self, learning_rate_and_kwargs):
        base_learning_rate, kwargs = learning_rate_and_kwargs
        self._global_step = tf.Variable(0, trainable=False)
        if kwargs:
            kwargs['decay_steps']=kwargs.pop('decay_epochs',10)*self.num_batches_in_epoch
            self.learning_rate = tf.train.exponential_decay(base_learning_rate, self._global_step, **kwargs)
        else:
            self.learning_rate = tf.constant(base_learning_rate)

    @property
    def cur_op_fn(self):
        return self._op_fn

    @cur_op_fn.setter
    def cur_op_fn(self, op_fn_and_kwargs):
        op_fn, kwargs = op_fn_and_kwargs
        if op_fn == "adam":
            op_cls = AdamOptimizer
        elif op_fn == "grad":
            op_cls = GradientDescentOptimizer
        elif op_fn == "adagrad":
            op_cls = AdagradOptimizer
        elif op_fn == "momentum":
            op_cls = MomentumOptimizer
        elif op_fn == "adadelta":
            op_cls = AdadeltaOptimizer
        elif op_fn == "rmsprop":
            op_cls = RMSPropOptimizer
        self._op_fn = op_cls(learning_rate=self.cur_learning_rate, **kwargs)

    @property
    def cur_objective_fn(self):
        return self._objective_fn

    @cur_objective_fn.setter
    def cur_objective_fn(self, objective_fn):
        self.objective_fn = objective_fn
        self._objective_fn = self.get_objective_fn(objective_fn)

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

    @property
    def regularization(self):
        if self.regularizer_type is not None:
            if self.regularizer_type == "L1":
                regularizer = tf.contrib.layers.l1_regularizer(scale=self.regularizer_constant)
            elif self.regularizer_type == "L2":
                regularizer = tf.contrib.layers.l2_regularizer(scale=self.regularizer_constant)
            else:
                raise ValueError("Regularizer Type {} Unrecognized".format(self.regularizer_type))
            reg_variables = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            return tf.contrib.layers.apply_regularization(regularizer, reg_variables)
        else:
            return tf.constant(0.0)


    @regularization.setter
    def regularization(self, regularizer_args):
        if regularizer_args:
            self.regularizer_type, self.regularizer_constant = regularizer_args
        else:
            self.regularizer_type = None
