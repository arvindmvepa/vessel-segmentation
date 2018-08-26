from job.base import Job

import matplotlib
matplotlib.use('Agg')

from network.drive import DriveNetwork
from dataset.drive import DriveDataset

class DriveJob(Job):

    def __init__(self, OUTPUTS_DIR_PATH="."):
        super(DriveJob, self).__init__(OUTPUTS_DIR_PATH=OUTPUTS_DIR_PATH)

    @property
    def dataset_cls(self):
        return DriveDataset

    @property
    def network_cls(self):
        return DriveNetwork