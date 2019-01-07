import matplotlib
matplotlib.use('Agg')

import matplotlib

matplotlib.use('Agg')
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, cohen_kappa_score, roc_auc_score, confusion_matrix, \
    roc_curve, auc
from job.base import Job

class JobWoMasks(Job):

    def __init__(self, OUTPUTS_DIR_PATH="."):
        super(JobWoMasks, self).__init__(OUTPUTS_DIR_PATH=OUTPUTS_DIR_PATH)

    def get_network_dict(self, network, input_data, train=True):
        net_dict = super(JobWoMasks, self).get_network_dict(network, input_data, train=train)
        net_dict.update({network.inputs: input_data[0], network.targets: input_data[1]})
        return net_dict

    @staticmethod
    def get_max_threshold_accuracy_image(results, neg_class_frac, pos_class_frac, targets):
        fprs, tprs, thresholds = roc_curve(targets.flatten(), results.flatten())
        list_fprs_tprs_thresholds = list(zip(fprs, tprs, thresholds))
        interval = 0.0001
        thresh_max = 0.0

        for i in np.arange(0.0, 1.0 + interval, interval):
            index = int(round((len(thresholds) - 1) * i, 0))
            fpr, tpr, threshold = list_fprs_tprs_thresholds[index]
            thresh_acc = (1 - fpr) * neg_class_frac + tpr * pos_class_frac
            if thresh_acc > thresh_max:
                thresh_max = thresh_acc
            i += 1
        return thresh_max