import tensorflow as tf

class Network(object):

    def __init__(self, wce_pos_weight=1, **kwargs):

        self.wce_pos_weight = wce_pos_weight