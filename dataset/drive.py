"""This is the file for the DriveDataset subclass"""

from dataset.dataset_w_masks import DatasetWMasks
from network.drive import DriveNetwork

class DriveDataset(DatasetWMasks):


    def __init__(self, **kwargs):
        super(DriveDataset, self).__init__(**kwargs)
        

    @property
    def network_cls(self):
        return DriveNetwork


    

