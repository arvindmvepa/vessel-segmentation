"""This is the file for the DRIVE network subclass"""

from network.small_network_w_masks import SmallNetworkWMasks

class DriveNetwork(SmallNetworkWMasks):

    IMAGE_HEIGHT = 584
    IMAGE_WIDTH = 565

    FIT_IMAGE_HEIGHT = 584
    FIT_IMAGE_WIDTH = 584
    
    def __init__(self, **kwargs):
        super(DriveNetwork, self).__init__(**kwargs)
