import matplotlib
matplotlib.use('Agg')

import matplotlib

matplotlib.use('Agg')
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, cohen_kappa_score, roc_auc_score, confusion_matrix, \
    roc_curve, auc
import os
from job.base import Job
from scipy.misc import imsave


class JobWMasks(Job):

    def __init__(self, OUTPUTS_DIR_PATH="."):
        super(JobWMasks, self).__init__(OUTPUTS_DIR_PATH=OUTPUTS_DIR_PATH)

    def get_network_dict(self, network, input_data, train=True):
        net_dict = super(JobWMasks, self).get_network_dict(network, input_data, train=train)
        net_dict.update({network.inputs: input_data[0], network.masks: input_data[1], network.targets: input_data[2]})
        return net_dict

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
        super(JobWMasks, self).save_data(prediction_flat, target_flat, mask_flat, timestamp, epoch_i)
        masks_path = os.path.join(self.OUTPUTS_DIR_PATH, "saved_masks")

        if not os.path.exists(masks_path):
            os.makedirs(masks_path)
        if not os.path.exists(os.path.join(masks_path, "mask.npy")):
            np.save(os.path.join(masks_path, "mask.npy"), mask_flat)

    def get_test_mask_flat(self, dataset):
        return dataset.test_masks.flatten()

    def get_val_mask_flat(self, dataset):
        return dataset.val_masks.flatten()

    @staticmethod
    def save_debug1(input_data, save_path):
        test1 = np.array(np.round(input_data[0][0,:,:]*255), dtype=np.uint8)
        test1_mask = np.array(np.round(input_data[1][0,:,:]*255), dtype=np.uint8)
        test1_target = np.array(np.round(input_data[2][0,:,:]*255), dtype=np.uint8)
        imsave(os.path.join(save_path, "test1.jpeg"), test1)
        imsave(os.path.join(save_path, "test1_mask.jpeg"), test1_mask)
        imsave(os.path.join(save_path, "test1_target.jpeg"), test1_target)

    @staticmethod
    def save_debug2(input_data, save_path):
        test2 = np.array(np.round(input_data[0][0,:,:,0]*255), dtype=np.uint8)
        test2_mask = np.array(np.round(input_data[1][0,:,:,0]*255), dtype=np.uint8)
        test2_target = np.array(np.round(input_data[2][0,:,:,0]*255), dtype=np.uint8)
        imsave(os.path.join(save_path, "test2.jpeg"), test2)
        imsave(os.path.join(save_path, "test2_mask.jpeg"), test2_mask)
        imsave(os.path.join(save_path, "test2_target.jpeg"), test2_target)

    @staticmethod
    def save_debug3(input_data, debug_data, save_path):
        test3 = np.array(np.round(input_data[0][0,:,:,0]*255), dtype=np.uint8)
        test3_mask = np.array(np.round(input_data[1][0,:,:,0]*255), dtype=np.uint8)
        test3_target = np.array(np.round(input_data[2][0,:,:,0]*255), dtype=np.uint8)
        imsave(os.path.join(save_path, "test3.jpeg"), test3)
        imsave(os.path.join(save_path, "test3_mask.jpeg"), test3_mask)
        imsave(os.path.join(save_path, "test3_target.jpeg"), test3_target)
        debug1 = debug_data[0,:,:,0]
        debug1 = np.array(np.round(debug1*255), dtype=np.uint8)
        imsave(os.path.join(save_path, "netowrk_input_debug1.jpeg"), debug1)