"""This is the file for the DRIVE network subclass"""

from network.small_network import SmallNetwork, SmallNetworkwKerasDecoder
from network.large_network import LargeNetwork


class DriveNetwork(SmallNetwork):

    IMAGE_HEIGHT = 584
    IMAGE_WIDTH = 565

    FIT_IMAGE_HEIGHT = 584
    FIT_IMAGE_WIDTH = 584
    
    def __init__(self, mask=True,**kwargs):
        super(DriveNetwork, self).__init__(mask=mask,**kwargs)


class DriveLargeNetwork(LargeNetwork):
    IMAGE_HEIGHT = 584
    IMAGE_WIDTH = 565

    FIT_IMAGE_HEIGHT = 584
    FIT_IMAGE_WIDTH = 584

    def __init__(self, mask=True, **kwargs):
        super(DriveNetwork, self).__init__(mask=mask, **kwargs)


class DriveCustomNetwork(SmallNetworkwKerasDecoder):
    IMAGE_HEIGHT = 584
    IMAGE_WIDTH = 565

    FIT_IMAGE_HEIGHT = 584
    FIT_IMAGE_WIDTH = 584

    def __init__(self, mask=True, **kwargs):
        super(DriveCustomNetwork, self).__init__(mask=mask, **kwargs)
