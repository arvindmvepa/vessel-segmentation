from network.retinal_wo_masks import RetinalWoMasksNetwork

class StareNetwork(RetinalWoMasksNetwork):
    IMAGE_WIDTH = 700
    IMAGE_HEIGHT = 605

    FIT_IMAGE_HEIGHT = 704
    FIT_IMAGE_WIDTH = 704

    def __init__(self, layers=None, skip_connections=True, **kwargs):
        super(StareNetwork, self).__init__(layers=layers, skip_connections=skip_connections, **kwargs)
