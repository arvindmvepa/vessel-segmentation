"""This is the file for the DsaJob subclass"""
from job.job_wo_masks import JobWoMasks

import matplotlib
matplotlib.use('Agg')

from dataset.dsa import DsaDataset
from network.dsa import DsaNetwork

class DsaJob(JobWoMasks):

    def __init__(self, OUTPUTS_DIR_PATH="."):
        super(DsaJob, self).__init__(OUTPUTS_DIR_PATH=OUTPUTS_DIR_PATH)

    @property
    def dataset_cls(self):
        return DsaDataset

    @property
    def network_cls(self):
        return DsaNetwork
