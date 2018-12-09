from network.small_network import SmallNetwork
from layers.conv_ops import Conv2d, ConvT2d
from layers.pool_ops import Pool2d, UnPool2d
from utilities.misc import update
from tensorflow.keras.applications import DenseNet, InceptionResNetV2, InceptionV3, MobileNet, MobileNetV2, NASNet, \
    ResNet50, ResNet101, ResNet152, ResNet50V2, ResNet101V2, ResNet152V2, ResNeXt50, ResNeXt101, VGG16, VGG19, Xception
import tensorflow as tf

class SmallNetworkwKerasDecoder(SmallNetwork):

    encoder_map = {"densenet": DenseNet, "incresv2": InceptionResNetV2, "incv3": InceptionV3, "mbnet": MobileNet,
                   "mbnetv2": MobileNetV2, "nasnet": NASNet, "resnet50": ResNet50, "resnet101": ResNet101,
                   "resnet152": ResNet152, "resnet50v2": ResNet50V2, "resnet101v2": ResNet101V2,
                   "resnet152v2": ResNet152V2, "resnext50": ResNeXt50, "resnext101": ResNeXt101, "vgg16": VGG16,
                   "vgg19": VGG19, "xception": Xception}

    def __init__(self, encoder_model_key="resnet50", layer_params=None, **kwargs):
        updated_layer_params = {"up_6": {"add_to_input": False},
                                "up_7": {"add_to_input": False},
                                "up_8": {"add_to_input": False}}
        if layer_params:
            layer_params = update(updated_layer_params, layer_params)
        else:
            layer_params = updated_layer_params
        super(SmallNetworkwKerasDecoder, self).__init__(encoder_model_key=encoder_model_key, layer_params=layer_params,
                                                        **kwargs)

    def init_encoder(self, encoder_model_key, encoder_layer_name, **kwargs):
        # have to multiply input image for 3 channels for ImageNet models
        self.inputs = tf.concat([self.inputs, self.inputs, self.inputs], axis=-1)
        # tensor has to be added to keras model so graph isn't duplicated
        # weights are hard-coded to be random, don't want to deal with pre-trained being clobbered during initialization
        # http://zachmoshe.com/2017/11/11/use-keras-models-with-tf.html
        base_model = self.encoder_map[self.encoder_model_key](weights=None, include_top=False, input_tensor=self.inputs)
        layers = {l.name: l.output for l in base_model.layers}
        self.encoder = layers[encoder_layer_name]



    def encode(self, **kwargs):
        return self.encoder
