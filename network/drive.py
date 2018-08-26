"""This is the file for the DRIVE network subclass"""

from network.retinal_w_masks import RetinalWMasksNetwork

class DriveNetwork(RetinalWMasksNetwork):

    # actual image dimensions 
    IMAGE_HEIGHT = 584
    IMAGE_WIDTH = 565

    # transformed input dimensions for network input
    FIT_IMAGE_HEIGHT = 584
    FIT_IMAGE_WIDTH = 584

    def __init__(self, layers=None, skip_connections=True, **kwargs):
        super(DriveNetwork, self).__init__(layers=layers, skip_connections=skip_connections, **kwargs)
