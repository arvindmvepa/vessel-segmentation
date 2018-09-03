"""This is the file for the DSA network subclass"""
from network.large_network import LargeNetwork


class DsaNetwork(LargeNetwork):

    IMAGE_WIDTH = 1024
    IMAGE_HEIGHT = 1024

    FIT_IMAGE_WIDTH = 1024
    FIT_IMAGE_HEIGHT = 1024

    def __init__(self, mask=False, **kwargs):
        super(DsaNetwork, self).__init__(mask=mask,**kwargs)