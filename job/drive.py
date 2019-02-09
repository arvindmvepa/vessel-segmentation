"""This is the file for the DriveJob subclass"""

from job.job_w_masks import JobWMasks

import matplotlib
matplotlib.use('Agg')

from network.drive import DriveNetwork, DriveLargeNetwork, DriveCustomNetwork
from dataset.drive import DriveDataset, DriveLargeDataset

class DriveJob(JobWMasks):

    def __init__(self,OUTPUTS_DIR_PATH="."):
        super(DriveJob, self).__init__(OUTPUTS_DIR_PATH=OUTPUTS_DIR_PATH)
        

    @property
    def dataset_cls(self):
        return DriveDataset

    @property
    def network_cls(self):
        return DriveNetwork


class DriveLargeJob(DriveJob):

    @property
    def dataset_cls(self):
        return DriveLargeDataset

    @property
    def network_cls(self):
        return DriveLargeNetwork


class DriveCustomJob(JobWMasks):
    def __init__(self, OUTPUTS_DIR_PATH="."):
        super(DriveCustomJob, self).__init__(OUTPUTS_DIR_PATH=OUTPUTS_DIR_PATH)

    @property
    def dataset_cls(self):
        return DriveDataset

    @property
    def network_cls(self):
        return DriveCustomNetwork
