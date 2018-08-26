from network.retinal_wo_masks import RetinalWoMasksNetwork

class ChaseNetwork(RetinalWoMasksNetwork):
    IMAGE_WIDTH = 999
    IMAGE_HEIGHT = 960

    FIT_IMAGE_WIDTH = 1024
    FIT_IMAGE_HEIGHT = 1024

    def __init__(self, layers=None, skip_connections=True, **kwargs):
        super(ChaseNetwork, self).__init__(layers=layers, skip_connections=skip_connections, **kwargs)