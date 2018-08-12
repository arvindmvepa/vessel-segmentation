from job.base import Job

import os
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


import datetime
from sklearn.metrics import precision_recall_fscore_support, cohen_kappa_score, roc_auc_score, confusion_matrix, \
    roc_curve, auc

from random import randint

from dataset.dsa import DsaDataset
from network.dsa import DsaNetwork

from utilities.output_ops import draw_results
import csv

# debugging

class DsaJob(Job):

    metrics = ("test set average weighted log loss","test set average unweighted log loss",
               "training set batch weighted log loss","training set batch unweighted log loss","auc","aucfpr10",
               "aucfpr05","aucfpr025","accuracy","precision","recall","specificity","f1_score","kappa","dt accuracy",
               "dt precision","dt recall","dt specificity","dt f1_score","dt kappa","threshold scores")

    def __init__(self, OUTPUTS_DIR_PATH="."):
        super(DsaJob, self).__init__(OUTPUTS_DIR_PATH=OUTPUTS_DIR_PATH)

    def train(self, gpu_device=None, decision_threshold=.75, tuning_constant=1.0, metrics_epoch_freq=1,
              viz_layer_epoch_freq=10, n_epochs=100, metrics_log="metrics_log.csv", loss_log="loss_log.txt",
              batch_size=1, num_image_plots=5, save_model=True, debug_net_output=True, **ds_kwargs):

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")

        dataset = DsaDataset(**ds_kwargs)
        pos_weight = dataset.get_tuned_pos_ce_weight(tuning_constant=tuning_constant, targets=dataset.train_targets)

        # initialize network object
        if gpu_device is not None:
            with tf.device(gpu_device):
                network = DsaNetwork(wce_pos_weight=pos_weight)
        else:
            network = DsaNetwork(wce_pos_weight=pos_weight)

        # create metrics log file
        metric_log_file_path = os.path.join(self.OUTPUTS_DIR_PATH, metrics_log)
        self.write_to_csv(sorted(DsaJob.metrics),metric_log_file_path)

        # create summary writer
        summary_writer = tf.summary.FileWriter('{}/{}/{}-{}'.format(self.OUTPUTS_DIR_PATH, 'logs', network.description,
                                                                    timestamp),
                                               graph=tf.get_default_graph())

        # create directories and subdirectories
        if save_model:
            os.makedirs(os.path.join(self.OUTPUTS_DIR_PATH, 'save', network.description, timestamp))

        if viz_layer_epoch_freq is not None:
            viz_layer_outputs_path_train, viz_layer_outputs_path_test = \
                self.create_viz_dirs(network,timestamp)

        config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            tf.train.Saver(tf.all_variables(), max_to_keep=None)

            for epoch_i in range(n_epochs):
                dataset.reset_batch_pointer()
                for batch_i in range(dataset.num_batches_in_epoch()):
                    start = time.time()
                    batch_num = epoch_i* dataset.num_batches_in_epoch() + batch_i
                    batch_inputs, batch_targets = dataset.next_batch()


                    if viz_layer_epoch_freq is not None and debug_net_output:
                        plt.imsave(os.path.join(viz_layer_outputs_path_train, "test1.jpeg"), batch_inputs[0])
                        plt.imsave(os.path.join(viz_layer_outputs_path_train, "test1_target.jpeg"),
                                   batch_targets[0])

                    # reshape array for tensorflow
                    batch_inputs = np.reshape(batch_inputs, (dataset.batch_size, batch_inputs.shape[1],
                                                             batch_inputs.shape[2], 1))
                    batch_targets = np.reshape(batch_targets, (dataset.batch_size, batch_targets.shape[1],
                                                               batch_targets.shape[2], 1))

                    if viz_layer_epoch_freq is not None and debug_net_output:
                        plt.imsave(os.path.join(viz_layer_outputs_path_train, "test2.jpeg"), batch_inputs[0,:,:,0])
                        plt.imsave(os.path.join(viz_layer_outputs_path_train, "test2_target.jpeg"),
                                   batch_targets[0,:,:,0])
                    cost, cost_unweighted, layer_outputs, debug1, acc, _ = sess.run([network.cost, network.cost_unweighted,
                                                                                     network.layer_outputs, network.debug1,
                                                                                     network.accuracy, network.train_op],
                                                                                    feed_dict={network.inputs: batch_inputs,
                                                                                               network.targets: batch_targets,
                                                                                               network.is_training: True})
                    end = time.time()
                    print('{}/{}, epoch: {}, cost: {}, cost unweighted: {}, batch time: {}, positive_weight: {}, accuracy: {}'.format(
                        batch_num, n_epochs * dataset.num_batches_in_epoch(), epoch_i, cost, cost_unweighted,
                        end - start, pos_weight, acc))

                    if viz_layer_epoch_freq is not None and debug_net_output:
                        plt.imsave(os.path.join(viz_layer_outputs_path_train, "test2.jpeg"), batch_inputs[0,:,:,0])
                        plt.imsave(os.path.join(viz_layer_outputs_path_train, "test2_target.jpeg"),
                                   batch_targets[0,:,:,0])
                        debug1 = debug1[0,:,:,0]
                        plt.imsave(os.path.join(viz_layer_outputs_path_train, "debug1.jpeg"), debug1)

                    if (epoch_i+1) % viz_layer_epoch_freq == 0 and batch_i == dataset.num_batches_in_epoch()-1:
                        self.create_viz_layer_output(layer_outputs, decision_threshold, viz_layer_outputs_path_train)

                    if (epoch_i + 1) % metrics_epoch_freq == 0 and batch_i == dataset.num_batches_in_epoch() - 1:
                        self.evaluate_on_test_set(metric_log_file_path, network, dataset, sess,
                                                  decision_threshold, epoch_i, timestamp,viz_layer_epoch_freq,
                                                  viz_layer_outputs_path_test, num_image_plots,summary_writer,
                                                  cost=cost, cost_unweighted=cost_unweighted)

    def evaluate_on_test_set(self, metrics_log_file_path, network, dataset, sess, decision_threshold, epoch_i, timestamp,
                             viz_layer_epoch_freq, viz_layer_outputs_path_test, num_image_plots, summary_writer,
                             **kwargs):

        metric_scores = dict()

        max_thresh_accuracy = 0.0
        test_cost = 0.0
        test_cost_unweighted = 0.0

        segmentation_results = np.zeros((dataset.test_targets.shape[0], dataset.test_targets.shape[1],
                                         dataset.test_targets.shape[2]))
        sample_test_image = randint(0, len(dataset.test_images) - 1)
        # get test results per image
        for i, (test_image, test_target) in enumerate(zip(dataset.test_images, dataset.test_targets)):
            test_cost_, test_cost_unweighted_, segmentation_result, layer_outputs = \
                sess.run([network.cost, network.cost_unweighted, network.segmentation_result,  network.layer_outputs],
                         feed_dict={network.inputs: np.reshape(test_image, (1, test_image.shape[0], test_image.shape[1],
                                                                            1)),
                                    network.targets: np.reshape(test_target, (1, test_target.shape[0],
                                                                              test_target.shape[1], 1)),
                                    network.is_training: False})
            # print('test {} : epoch: {}, cost: {}, cost unweighted: {}'.format(i,epoch_i,test_cost,test_cost_unweighted))

            segmentation_result = segmentation_result[0, :, :, 0]
            segmentation_results[i,:,:] = segmentation_result

            test_cost += test_cost_
            test_cost_unweighted += test_cost_unweighted_

            _, test_neg_class_frac, test_pos_class_frac = dataset.get_inverse_pos_freq(targets=dataset.test_targets[i])

            # calculate max threshold accuracy per test image
            thresh_max = self.get_max_threshold_accuracy_image(segmentation_results, dataset.test_targets,
                                                               test_neg_class_frac, test_pos_class_frac)
            max_thresh_accuracy += thresh_max

            if i == sample_test_image and (epoch_i + 1) % viz_layer_epoch_freq == 0:
                self.create_viz_layer_output(layer_outputs, decision_threshold, viz_layer_outputs_path_test)

        # combine test results to produce overall metric scores
        max_thresh_accuracy = max_thresh_accuracy / len(dataset.test_images)
        test_cost = test_cost / len(dataset.test_images)
        test_cost_unweighted = test_cost_unweighted / len(dataset.test_images)

        prediction_flat = segmentation_results.flatten()
        target_flat =np.round(dataset.test_targets.flatten())

        # save target (if files don't exist)
        targets_path = os.path.join(self.OUTPUTS_DIR_PATH, "saved_targets")
        preds_path = os.path.join(self.OUTPUTS_DIR_PATH, "saved_preds")

        if not os.path.exists(targets_path):
            os.makedirs(targets_path)
        if not os.path.exists(preds_path):
            os.makedirs(preds_path)

        if not os.path.exists(os.path.join(targets_path, "target.npy")):
            np.save(os.path.join(targets_path,"target.npy"), target_flat)

        # save prediction array for ensemble processing potentially
        file_name = timestamp + "_" + str(epoch_i)
        np.save(os.path.join(preds_path,file_name), prediction_flat)


        # produce AUCROC score with map
        auc_score = roc_auc_score(target_flat, prediction_flat)

        # produce auc_score curve thresholded at different FP poinnts
        fprs, tprs, thresholds = roc_curve(target_flat, prediction_flat)
        np_fprs, np_tprs, np_thresholds = np.array(fprs).flatten(), np.array(tprs).flatten(), np.array(
            thresholds).flatten()

        fpr_10 = np_fprs[np.where(np_fprs < .10)]
        tpr_10 = np_tprs[0:len(fpr_10)]

        fpr_05 = np_fprs[np.where(np_fprs < .05)]
        tpr_05 = np_tprs[0:len(fpr_05)]

        fpr_025 = np_fprs[np.where(np_fprs < .025)]
        tpr_025 = np_tprs[0:len(fpr_025)]

        auc_10_fpr = auc(fpr_10, tpr_10)
        auc_05_fpr = auc(fpr_05, tpr_05)
        auc_025_fpr = auc(fpr_025, tpr_025)

        # produce accuracy at different decision thresholds
        list_fprs_tprs_thresholds = list(zip(fprs, tprs, thresholds))
        interval = 0.05

        threshold_scores = [max_thresh_accuracy]
        for i in np.arange(0, 1.0 + interval, interval):
            index = int(round((len(thresholds) - 1) * i, 0))
            fpr, tpr, threshold = list_fprs_tprs_thresholds[index]
            thresh_acc = (1 - fpr) * test_neg_class_frac + tpr * test_pos_class_frac
            threshold_scores.append((threshold, thresh_acc, tpr, 1-fpr))

        # produce metrics based on predictions given by decisions thresholded at .5
        rounded_prediction_flat = np.round(prediction_flat)
        (precision, recall, fbeta_score, _) = precision_recall_fscore_support(target_flat,
                                                                              rounded_prediction_flat,
                                                                              average='binary')

        tn, fp, fn, tp = confusion_matrix(target_flat, rounded_prediction_flat).ravel()
        kappa = cohen_kappa_score(target_flat, rounded_prediction_flat)
        acc = float(tp + tn) / float(tp + tn + fp + fn)
        specificity = float(tn) / float(tn + fp)

        # produce metrics based on predictions given by decision_threshold
        thresh_prediction_flat = (prediction_flat > decision_threshold).astype(int)

        (r_precision, r_recall, r_fbeta_score, _) = precision_recall_fscore_support(target_flat,
                                                                              thresh_prediction_flat,
                                                                              average='binary')

        r_tn, r_fp, r_fn, r_tp = confusion_matrix(target_flat, thresh_prediction_flat).ravel()
        r_kappa = cohen_kappa_score(target_flat, thresh_prediction_flat)
        r_acc = float(r_tp + r_tn) / float(r_tp + r_tn + r_fp + r_fn)
        r_specificity = float(r_tn) / float(r_tn + r_fp)

        metric_scores["test set average weighted log loss"] = test_cost
        metric_scores["test set average unweighted log loss"] = test_cost_unweighted
        metric_scores["training set batch weighted log loss"] = kwargs["cost"]
        metric_scores["training set batch unweighted log loss"] = kwargs["cost_unweighted"]

        metric_scores["auc"] = auc_score
        metric_scores["aucfpr10"] = auc_10_fpr
        metric_scores["aucfpr05"] = auc_05_fpr
        metric_scores["aucfpr025"] = auc_025_fpr

        metric_scores["accuracy"] = acc
        metric_scores["precision"] = precision
        metric_scores["recall"] = recall
        metric_scores["specificity"] = specificity
        metric_scores["f1_score"] = fbeta_score
        metric_scores["kappa"] = kappa

        metric_scores["dt accuracy"] = r_acc
        metric_scores["dt precision"] = r_precision
        metric_scores["dt recall"] = r_recall
        metric_scores["dt specificity"] = r_specificity
        metric_scores["dt f1_score"] = r_fbeta_score
        metric_scores["dt kappa"] = r_kappa

        metric_scores["threshold scores"] = threshold_scores

        # save metric results to log
        self.write_to_csv([metric_scores[key] for key in sorted(metric_scores.keys())], metrics_log_file_path)

        # produce image plots
        test_plot_buf = draw_results(dataset.test_images[:num_image_plots],
                                     dataset.test_targets[:num_image_plots], segmentation_results[:num_image_plots,:,:],
                                     acc, network, epoch_i, num_image_plots, os.path.join(self.OUTPUTS_DIR_PATH,
                                                                                          self.IMAGE_PLOT_DIR),
                                     decision_threshold)

        image = tf.image.decode_png(test_plot_buf.getvalue(), channels=4)
        image = tf.expand_dims(image, 0)
        image_summary_op = tf.summary.image("plot", image)
        image_summary = sess.run(image_summary_op)
        summary_writer.add_summary(image_summary)


    @staticmethod
    def get_max_threshold_accuracy_image(results, targets, neg_class_frac, pos_class_frac):
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