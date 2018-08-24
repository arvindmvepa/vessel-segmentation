"""This is the file for the base network class"""
import tensorflow as tf

class Network(object):

    def __init__(self, wce_pos_weight=1, layers=None, **kwargs):

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
            net = layer.create_layer_reversed(net, prev_layer=self.layers[layer.name])
            self.layer_outputs.append(net)

        self.net_output(net)

    def net_output(self, net):
        """This method produces the network output"""
        raise NotImplementedError("Not Implemented")