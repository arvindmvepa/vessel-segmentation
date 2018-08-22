from network.small_network import SmallNetwork

class StareNetwork(SmallNetwork):

    IMAGE_WIDTH = 700
    IMAGE_HEIGHT = 605

    FIT_IMAGE_HEIGHT = 704
    FIT_IMAGE_WIDTH = 704

    def __init__(self, mask=True, **kwargs):
        super(StareNetwork, self).__init__(mask=mask, **kwargs)
