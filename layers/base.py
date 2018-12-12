import tensorflow as tf

class Layer(object):
    def __init__(self, center=None, **kwargs):
        super(Layer, self).__init__(**kwargs)
        self.center = center

    def create_layer(self, input, **kwargs):
        pass

    def zero_center_output(self, output, center):
        if self.center is not None:
            if self.center:
                print("zero center activation override")
                output = output - tf.reduce_mean(output)
            else:
                print("don't zero center activation override")
        else:
            if center:
                print("zero center activation normal")
                output = output - tf.reduce_mean(output)
            else:
                print("don't zero center activation normal")
        return output

    def get_description(self):
        pass

