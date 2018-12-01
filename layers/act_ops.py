import tensorflow as tf

from utilities.activations import lrelu

class ReLUOp():
    def __init__(self):
        pass

    def create_layer(self, input, **kwargs):
        return tf.nn.relu(input)

    def get_description(self):
        return "R"

class LReLUOp():
    def __init__(self, act_leak_prob):
        self.act_leak_prob = act_leak_prob

    def create_layer(self, input, **kwargs):
        return lrelu(input, self.act_leak_prob)

    def get_description(self):
        return "LR"