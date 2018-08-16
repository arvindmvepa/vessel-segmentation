from job.job_wo_masks import JobWoMasks

import matplotlib
matplotlib.use('Agg')

from dataset.chase import ChaseDataset
from network.chase import ChaseNetwork

class ChaseJob(JobWoMasks):

    def __init__(self, OUTPUTS_DIR_PATH="."):
        super(ChaseJob, self).__init__(OUTPUTS_DIR_PATH=OUTPUTS_DIR_PATH)

    @property
    def dataset_cls(self):
        return ChaseDataset

    @property
    def network_cls(self):
        return ChaseNetwork
