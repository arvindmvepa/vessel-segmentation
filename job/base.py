import multiprocessing
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

class Job(object):

    def __init__(self, OUTPUTS_DIR_PATH="."):
        if not os.path.exists(OUTPUTS_DIR_PATH):
            os.makedirs(OUTPUTS_DIR_PATH)
        self.OUTPUTS_DIR_PATH = OUTPUTS_DIR_PATH

    def train(self, **kwargs):
        raise NotImplementedError("Method Not Implemented")

    def run_single_model(self, **kwargs):
        p = multiprocessing.Process(target=self.train, kwargs=kwargs)
        p.start()
        p.join()

    def run_ensemble(self, count=10.0, decision_thresh=.75, wce_dist=True, wce_start_tuning_constant=.5,
                     wce_end_tuning_constant=2.0, wce_tuning_constants=None, wce_tuning_constant=1.0,
                     test_metrics_freq=200, layer_output_freq=1000, end=2000, job_dir=None, test_metrics_file=None,
                     log_loss_file=None):

        if wce_dist and wce_tuning_constants is not None:
            assert len(wce_tuning_constants) == count

        # uniformly distribute tuning constants used for ensemble if wce_tuning_constants not provided
        elif wce_dist:
            if count == 1:
                wce_tuning_constants = [wce_start_tuning_constant]
            else:
                interval = (wce_end_tuning_constant - wce_start_tuning_constant) / float(count - 1)
                wce_tuning_constants = list(
                    np.arange(wce_start_tuning_constant, wce_end_tuning_constant + interval, interval))

        for i in range(count):
            # select from tuning constants if provided, else use default value
            if wce_dist:
                wce_tuning_constant = wce_tuning_constants[i]

            kwargs = {"decision_thresh": decision_thresh, "wce_tuning_constant": wce_tuning_constant,
                      "test_metrics_freq": test_metrics_freq, "layer_output_freq": layer_output_freq, "end": end,
                      "job_dir": job_dir, "test_metrics_file": test_metrics_file, "log_loss_file": log_loss_file}

            p = multiprocessing.Process(target=self.train, kwargs=kwargs)
            p.start()
            p.join()

    def create_viz_dirs(self, network, timestamp, debug_net_output=False):
        viz_layer_outputs_path = os.path.join(self.OUTPUTS_DIR_PATH, 'viz_layer_outputs', network.description, timestamp)
        os.makedirs(viz_layer_outputs_path)
        viz_layer_outputs_path_train = os.path.join(viz_layer_outputs_path, "train")
        os.makedirs(viz_layer_outputs_path_train)
        viz_layer_outputs_path_test = os.path.join(viz_layer_outputs_path, "test")
        os.makedirs(viz_layer_outputs_path_test)
        for i, _ in enumerate(network.layers):
            viz_layer_output_path_train = os.path.join(viz_layer_outputs_path_train, str(i))
            os.makedirs(viz_layer_output_path_train)
            viz_layer_output_path_test = os.path.join(viz_layer_outputs_path_test, str(i))
            os.makedirs(viz_layer_output_path_test)
        viz_layer_mask1_output_path_train = os.path.join(viz_layer_outputs_path_train, "mask1")
        os.makedirs(viz_layer_mask1_output_path_train)
        viz_layer_mask2_output_path_train = os.path.join(viz_layer_outputs_path_train, "mask2")
        os.makedirs(viz_layer_mask2_output_path_train)
        if debug_net_output:
            viz_layer_debug1_output_path_train = os.path.join(viz_layer_outputs_path_train, "debug1")
            os.makedirs(viz_layer_debug1_output_path_train)
            viz_layer_debug2_output_path_train = os.path.join(viz_layer_outputs_path_train, "debug2")
            os.makedirs(viz_layer_debug2_output_path_train)
        return viz_layer_outputs_path_train, viz_layer_outputs_path_test

    def create_viz_layer_output_train(self, threshold, layer_outputs, output_path):
        for j, layer_output in enumerate(layer_outputs):
            for k in range(layer_output.shape[3]):
                channel_output = layer_output[0, :, :, k]
                plt.imsave(os.path.join(os.path.join(output_path, str(j + 1)),
                                        "channel_" + str(k) + ".jpeg"), channel_output)
                if j == 0:
                    channel_output[np.where(channel_output > threshold)] = 1
                    channel_output[np.where(channel_output <= threshold)] = 0
                    plt.imsave(os.path.join(os.path.join(output_path, "mask1"),
                                            "channel_" + str(k) + ".jpeg"), channel_output)
                if j == len(layer_outputs) - 1:
                    channel_output[np.where(channel_output > threshold)] = 1
                    channel_output[np.where(channel_output <= threshold)] = 0
                    plt.imsave(os.path.join(os.path.join(output_path, "mask2"),
                                            "channel_" + str(k) + ".jpeg"), channel_output)

    def create_viz_layer_output_test(self, threshold, layer_outputs, output_path, **kwargs):
        raise NotImplementedError("Method Not Implemented")

    @staticmethod
    def get_max_threshold_accuracy_image(targets, results, masks, neg_class_frac, pos_class_frac):
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

    def __call__(self):
        pass