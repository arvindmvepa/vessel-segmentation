from network.large_network_w_masks import LargeNetworkWMasks

class ChaseNetwork(LargeNetworkWMasks):

    IMAGE_HEIGHT = 960
    IMAGE_WIDTH = 999

    FIT_IMAGE_WIDTH = 1024
    FIT_IMAGE_HEIGHT = 1024

    def __init__(self, **kwargs):
        super(ChaseNetwork, self).__init__(**kwargs)