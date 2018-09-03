"""This is the file for the DSA network subclass"""
from network.large_network_wo_masks import LargeNetworkWoMasks


class DsaNetwork(LargeNetworkWoMasks):

    IMAGE_WIDTH = 1024
    IMAGE_HEIGHT = 1024

    FIT_IMAGE_WIDTH = 1024
    FIT_IMAGE_HEIGHT = 1024

    def __init__(self, **kwargs):
        super(DsaNetwork, self).__init__(**kwargs)