from network.large_network import LargeNetwork

class ChaseNetwork(LargeNetwork):

    IMAGE_HEIGHT = 960
    IMAGE_WIDTH = 999

    FIT_IMAGE_WIDTH = 1024
    FIT_IMAGE_HEIGHT = 1024

    def __init__(self, mask=True, **kwargs):
        super(ChaseNetwork, self).__init__(mask=True,**kwargs)