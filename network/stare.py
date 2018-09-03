from network.small_network_w_masks import SmallNetworkWMasks

class StareNetwork(SmallNetworkWMasks):
    IMAGE_WIDTH = 700
    IMAGE_HEIGHT = 605

    FIT_IMAGE_HEIGHT = 704
    FIT_IMAGE_WIDTH = 704

    def __init__(self, **kwargs):
        super(StareNetwork, self).__init__(**kwargs)
