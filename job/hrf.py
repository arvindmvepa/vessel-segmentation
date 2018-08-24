from job.base import Job
import matplotlib
matplotlib.use('Agg')
import numpy as np
import os
from network.hrf import HRFNetwork
from dataset.hrf import HRFDataset
from sklearn.metrics import precision_recall_fscore_support, cohen_kappa_score, roc_auc_score, confusion_matrix, \
    roc_curve, auc


class HRFJob(Job):

    def __init__(self, OUTPUTS_DIR_PATH="."):
        super(HRFJob, self).__init__(OUTPUTS_DIR_PATH=OUTPUTS_DIR_PATH)

    def get_network_dict(self, network, input_data, train=True):
        net_dict = super(HRFJob, self).get_network_dict(network, input_data, train=True)
        net_dict.update({network.inputs: input_data[0], network.masks: input_data[1], network.targets: input_data[2]})
        return net_dict

    @property
    def dataset_cls(self):
        return HRFDataset

    @property
    def network_cls(self):
        return HRFNetwork

    @staticmethod
    def get_max_threshold_accuracy_image(results, neg_class_frac, pos_class_frac, masks, targets):
        fprs, tprs, thresholds = roc_curve(targets.flatten(), results.flatten(), sample_weight=masks.flatten())
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

    def save_data(self, prediction_flat, target_flat, mask_flat, timestamp, epoch_i):
        super(HRFJob, self).save_data(prediction_flat, target_flat, mask_flat, timestamp, epoch_i)
        masks_path = os.path.join(self.OUTPUTS_DIR_PATH, "saved_masks")

        if not os.path.exists(masks_path):
            os.makedirs(masks_path)
        if not os.path.exists(os.path.join(masks_path, "mask.npy")):
            np.save(os.path.join(masks_path, "mask.npy"), mask_flat)

    def get_test_mask_flat(self, dataset):
        return dataset.test_masks.flatten()
