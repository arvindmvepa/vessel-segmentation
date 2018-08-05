import tensorflow as tf

class Network(object):

    def __init__(self, wce_pos_weight=1, layers=None, IMAGE_WIDTH=600, IMAGE_HEIGHT=600, FIT_IMAGE_WIDTH=600,
                 FIT_IMAGE_HEIGHT=600, IMAGE_CHANNELS=1, **kwargs):

        if layers == None :
            raise ValueError("No Layers Defined.")
        self.inputs = tf.placeholder(tf.float32, [None, FIT_IMAGE_WIDTH, FIT_IMAGE_HEIGHT, IMAGE_CHANNELS],
                                     name='inputs')
        self.targets = tf.placeholder(tf.float32, [None, IMAGE_WIDTH, IMAGE_HEIGHT, 1], name='targets')
        self.is_training = tf.placeholder_with_default(False, [], name='is_training')
        self.wce_pos_weight = wce_pos_weight
        self.layer_outputs = []
        for layer in layers:
            self.layer_outputs.append(tf.placeholder(tf.float32, [None, None, None, 1], name=layer.name+'_layer_outputs'))
        self.description = ""
        self.layers = {}

        # ENCODER
        for i, layer in enumerate(len(layers)):
            self.layers[layer.name] = net = layer.create_layer(net)
            self.description += "{}".format(layer.get_description())
            self.layer_outputs[i]=net

        print("Number of layers: ", len(layers))
        print("Current input shape: ", net.get_shape())

        layers.reverse()

        # DECODER
        for i, layer in enumerate(len(layers)):
            net = layer.create_layer_reversed(net, prev_layer=self.layers[layer.name])
            self.layer_outputs[len(layers)+i] = net

        net = tf.image.resize_image_with_crop_or_pad(net, IMAGE_WIDTH, IMAGE_HEIGHT)
        self.process_network_output(net)

    def process_network_output(self, net):
        raise NotImplementedError("Not Implemented")
