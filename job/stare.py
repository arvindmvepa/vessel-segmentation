from job.job_wo_masks import JobWoMasks

import matplotlib
matplotlib.use('Agg')
import numpy as np

from sklearn.metrics import precision_recall_fscore_support, cohen_kappa_score, roc_auc_score, confusion_matrix, \
    roc_curve, auc

from dataset.stare import StareDataset
from network.stare import StareNetwork

class StareJob(JobWoMasks):

    def __init__(self, OUTPUTS_DIR_PATH="."):
        super(StareJob, self).__init__(OUTPUTS_DIR_PATH=OUTPUTS_DIR_PATH)

    @property
    def dataset_cls(self):
        return StareDataset

    @property
    def network_cls(self):
        return StareNetwork