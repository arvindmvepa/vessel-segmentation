from job.base import Job
import os
import time
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import datetime
from sklearn.metrics import precision_recall_fscore_support, cohen_kappa_score, roc_auc_score, confusion_matrix, \
    roc_curve, auc
from random import randint
from dataset.stare import StareDataset
from network.stare import StareNetwork
from utilities.output_ops import draw_results
import csv

class StareJob(Job):

    metrics = ("cost","cost unweighted","accuracy","recall","specificity","precision","f1score","kappa","auc",
               "auc10fpr","auc5fpr","auc2.5fpr","class balance","threshold scores")

    def __init__(self, OUTPUTS_DIR_PATH="."):
        super(StareJob, self).__init__(OUTPUTS_DIR_PATH=OUTPUTS_DIR_PATH)

    def train(self, gpu_device=None, decision_threshold=.75, tuning_constant=1.0, metrics_epoch_freq=1,
              viz_layer_epoch_freq=10, n_epochs=100, metrics_log="metrics_log.txt", loss_log="loss_log.txt",
              batch_size=1, num_image_plots=5, save_model=True, debug_net_output=False, **ds_kwargs):

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")

        dataset = StareDataset(**ds_kwargs)
        pos_weight = dataset.get_tuned_pos_ce_weight(tuning_constant=tuning_constant, masks=dataset.test_masks,
                                                     targets=dataset.train_targets)

        # initialize network object
        if gpu_device is not None:
            with tf.device(gpu_device):
                network = StareNetwork(wce_pos_weight=pos_weight)
        else:
            network = StareNetwork(wce_pos_weight=pos_weight)

        # create summary writer
        summary_writer = tf.summary.FileWriter('{}/{}/{}-{}'.format(self.OUTPUTS_DIR_PATH, 'logs', network.description, timestamp),
                                               graph=tf.get_default_graph())

        # create directories and subdirectories
        if save_model:
            os.makedirs(os.path.join(self.OUTPUTS_DIR_PATH, 'save', network.description, timestamp))

        if viz_layer_epoch_freq is not None:
            viz_layer_outputs_path_train, viz_layer_outputs_path_test = \
                self.create_viz_dirs(network,timestamp,debug_net_output)

        config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            tf.train.Saver(tf.global_variables(), max_to_keep=None)

            for epoch_i in range(n_epochs):
                dataset.reset_batch_pointer()
                for batch_i in range(dataset.num_batches_in_epoch()):
                    start = time.time()
                    batch_num = epoch_i* dataset.num_batches_in_epoch() + batch_i
                    batch_inputs, batch_masks, batch_targets = dataset.next_batch()


                    if viz_layer_epoch_freq is not None and debug_net_output:
                        plt.imsave(os.path.join(viz_layer_outputs_path_train, "test1.jpeg"), batch_inputs[0])
                        plt.imsave(os.path.join(viz_layer_outputs_path_train, "test1_target.jpeg"),
                                   batch_targets[0])

                    # reshape array for tensorflow
                    batch_inputs = np.reshape(batch_inputs, (dataset.batch_size, batch_inputs.shape[1],
                                                             batch_inputs.shape[2], 1))
                    batch_masks = np.reshape(batch_masks, (dataset.batch_size, batch_masks.shape[1],
                                                           batch_masks.shape[2], 1))
                    batch_targets = np.reshape(batch_targets, (dataset.batch_size, batch_targets.shape[1],
                                                               batch_targets.shape[2], 1))

                    if viz_layer_epoch_freq is not None and debug_net_output:
                        plt.imsave(os.path.join(viz_layer_outputs_path_train, "test2.jpeg"), batch_inputs[0,:,:,0])
                        plt.imsave(os.path.join(viz_layer_outputs_path_train, "test2_target.jpeg"),
                                   batch_targets[0,:,:,0])

                    cost, cost_unweighted, layer_outputs, debug1 = sess.run([network.cost,
                                                                                        network.cost_unweighted,
                                                                           network.layer_outputs, network.debug1],
                                       feed_dict={network.inputs: batch_inputs, network.masks: batch_masks,
                                                  network.targets: batch_targets, network.is_training: True})
                    end = time.time()
                    print('{}/{}, epoch: {}, cost: {}, cost unweighted: {}, batch time: {}, positive_weight: {}'.format(
                        batch_num, n_epochs * dataset.num_batches_in_epoch(), epoch_i, cost, cost_unweighted,
                        end - start, pos_weight))

                    if viz_layer_epoch_freq is not None and debug_net_output:
                        plt.imsave(os.path.join(viz_layer_outputs_path_train, "test2.jpeg"), batch_inputs[0,:,:,0])
                        plt.imsave(os.path.join(viz_layer_outputs_path_train, "test2_target.jpeg"),
                                   batch_targets[0,:,:,0])
                        debug1 = debug1[0,:,:,0]
                        plt.imsave(os.path.join(viz_layer_outputs_path_train, "debug1.jpeg"), debug1)

                    if (epoch_i+1) % viz_layer_epoch_freq == 0 and batch_i == dataset.num_batches_in_epoch()-1:
                        self.create_viz_layer_output_train(decision_threshold, layer_outputs,
                                                           viz_layer_outputs_path_train)

                    if (epoch_i + 1) % metrics_epoch_freq == 0 and batch_i == dataset.num_batches_in_epoch() - 1:
                        self.evaluate_on_test_set(network, dataset, sess, decision_threshold, epoch_i, timestamp,
                                                  viz_layer_epoch_freq, viz_layer_outputs_path_test, num_image_plots,
                                                  summary_writer, cost=cost, cost_unweighted=cost_unweighted)

    def evaluate_on_test_set(self, network, dataset, sess, decision_threshold, epoch_i, timestamp,
                             viz_layer_epoch_freq, viz_layer_outputs_path_test, num_image_plots, summary_writer,
                             **kwargs):

        metric_scores = dict()

        max_thresh_accuracy = 0.0
        test_cost = 0.0
        test_cost_unweighted = 0.0

        mask_array = np.zeros((self.test_masks.shape[0], self.test_masks.shape[1], self.test_masks.shape[2]))
        target_array = np.zeros((self.test_targets.shape[0], self.test_targets.shape[1],
                                 self.test_targets.shape[2]))
        segmentation_results = np.zeros((self.test_targets.shape[0], self.test_targets.shape[1],
                                     self.test_targets.shape[2]))
        sample_test_image = randint(0, len(self.test_images) - 1)

        # get test results per image
        for i in range(len(self.test_images)):
            test_cost_, test_cost_unweighted_, segmentation_result, layer_outputs = \
                sess.run([network.cost, network.cost_unweighted, network.segmentation_result,  network.layer_outputs],
                         feed_dict={network.inputs: self.test_images[i:(i + 1)],
                                    network.masks: self.test_masks[i:(i + 1)],
                                    network.targets: self.test_targets[i:(i + 1)],
                                    network.is_training: False})

            masks = masks[0, :, :, 0]
            segmentation_result = segmentation_result[0, :, :, 0]
            targets = targets[0, :, :, 0]

            mask_array[i] = masks
            target_array[i] = targets
            segmentation_results[i] = segmentation_result

            test_cost += test_cost_
            test_cost_unweighted += test_cost_unweighted_

            _, test_neg_class_frac, test_pos_class_frac = \
                self.get_inverse_pos_freq(masks=dataset.test_masks[i], targets=dataset.test_targets[i])

            # calculate max threshold accuracy per test image
            thresh_max = self.get_max_threshold_accuracy_image(masks, targets, segmentation_result, test_neg_class_frac,
                                                               test_pos_class_frac)
            max_thresh_accuracy += thresh_max

            if i == sample_test_image and (epoch_i + 1) % viz_layer_epoch_freq == 0:
                self.create_viz_layer_output_test(decision_threshold, layer_outputs, viz_layer_outputs_path_test,
                                                  self.test_targets[i:(i + 1)], self.test_masks[i:(i + 1)])

        # combine test results to produce overall metric scores
        max_thresh_accuracy = max_thresh_accuracy / len(self.test_images)
        test_cost = test_cost / len(self.test_images)
        test_cost_unweighted = test_cost_unweighted / len(self.test_images)

        mask_flat = mask_array.flatten()
        prediction_flat = segmentation_results.flatten()
        target_flat =np.round(target_array.flatten())

        # save mask and target (if files don't exist)
        masks_path = os.path.join(self.OUTPUTS_DIR_PATH, "saved_masks")
        targets_path = os.path.join(self.OUTPUTS_DIR_PATH, "saved_targets")
        preds_path = os.path.join(self.OUTPUTS_DIR_PATH, "saved_preds")

        if not os.path.exists(os.path.join(masks_path, "mask.npy")):
            np.save(os.path.join(masks_path, "mask.npy"), mask_flat)
        if not os.path.exists(os.path.join(targets_path, "target.npy")):
            np.save(os.path.join(targets_path,"target.npy"), target_flat)

        # save prediction array for ensemble processing potentially
        file_name = timestamp + "_" + str(epoch_i)
        np.save(os.path.join(preds_path,file_name), prediction_flat)

        # produce AUCROC score with map
        auc_score = roc_auc_score(target_flat, prediction_flat, sample_weight=mask_flat)

        # produce auc_score curve thresholded at different FP poinnts
        fprs, tprs, thresholds = roc_curve(target_flat, prediction_flat, sample_weight=mask_flat)
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
                                                                              average='binary',
                                                                              sample_weight=mask_flat)

        tn, fp, fn, tp = confusion_matrix(target_flat, rounded_prediction_flat, sample_weight=mask_flat).ravel()
        kappa = cohen_kappa_score(target_flat, rounded_prediction_flat, sample_weight=mask_flat)
        acc = (tp + tn) / (tp + tn + fp + fn)
        specificity = tn / (tn + fp)

        # produce metrics based on predictions given by decision_threshold
        thresh_prediction_flat = (prediction_flat > decision_threshold).astype(int)

        (r_precision, r_recall, r_fbeta_score, _) = precision_recall_fscore_support(target_flat,
                                                                              thresh_prediction_flat,
                                                                              average='binary',
                                                                              sample_weight=mask_flat)

        r_tn, r_fp, r_fn, r_tp = confusion_matrix(target_flat, thresh_prediction_flat, sample_weight=mask_flat).ravel()
        r_kappa = cohen_kappa_score(target_flat, thresh_prediction_flat, sample_weight=mask_flat)
        r_acc = (r_tp + r_tn) / (r_tp + r_tn + r_fp + r_fn)
        r_specificity = r_tn / (r_tn + r_fp)

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

        # produce image plots
        test_plot_buf = draw_results(self.test_images[:num_image_plots],
                                     self.test_targets[:num_image_plots], segmentation_results[:num_image_plots,:,:],
                                     acc, network, epoch_i, decision_threshold)

        image = tf.image.decode_png(test_plot_buf.getvalue(), channels=4)
        image = tf.expand_dims(image, 0)
        image_summary_op = tf.summary.image("plot", image)
        image_summary = sess.run(image_summary_op)
        summary_writer.add_summary(image_summary)



    def create_csv(self, metrics, metrics_file):
        with open(metrics_file, 'wb') as csvfile:
            csvfile.writerow(metrics)

    def write_to_csv(self, metrics, metrics_file):
        with open(metrics_file, 'wb') as csvfile:
            csvfile.writerow(metrics)


    def create_viz_layer_output_test(self, threshold, layer_outputs, output_path, test_target, test_mask):
        top_pad = int((DriveNetwork.FIT_IMAGE_HEIGHT - DriveNetwork.IMAGE_HEIGHT) / 2)
        bot_pad = (DriveNetwork.FIT_IMAGE_HEIGHT - DriveNetwork.IMAGE_HEIGHT) - top_pad
        left_pad = int((DriveNetwork.FIT_IMAGE_WIDTH - DriveNetwork.IMAGE_WIDTH) / 2)
        right_pad = (DriveNetwork.FIT_IMAGE_WIDTH - DriveNetwork.IMAGE_WIDTH) - left_pad

        target = test_target
        target = np.reshape(target, (DriveNetwork.IMAGE_HEIGHT, DriveNetwork.IMAGE_WIDTH))
        invert_target = np.invert(target)

        mask = test_mask
        mask = np.reshape(mask, (DriveNetwork.IMAGE_HEIGHT, DriveNetwork.IMAGE_WIDTH))
        target = np.multiply(target, mask)
        invert_target = np.multiply(invert_target, mask)

        channel_list = []
        total_pos = 0
        total_neg = 0

        for j in range(len(layer_outputs)):
            layer_output = layer_outputs[j]
            for k in range(layer_output.shape[3]):
                channel_output = layer_output[0, :, :, k]
                plt.imsave(os.path.join(os.path.join(output_path, str(j + 1)),
                                        "channel_" + str(k) + ".jpeg"), channel_output)
                if j == 0:
                    input = np.reshape(channel_output, (DriveNetwork.FIT_IMAGE_HEIGHT,
                                                        DriveNetwork.FIT_IMAGE_WIDTH))
                    input_crop = input[top_pad:DriveNetwork.FIT_IMAGE_HEIGHT - bot_pad,
                                 left_pad:DriveNetwork.FIT_IMAGE_WIDTH - right_pad]
                    invert_input_crop = 1 - input_crop

                    results_ = np.multiply(input_crop, target)
                    results_invert = np.multiply(invert_input_crop, invert_target)

                    sum = np.sum(results_)
                    sum_neg = np.sum(results_invert)

                    total_pos += sum
                    total_neg += sum_neg

                    channel_list.append((sum, input_crop))

                    channel_output[np.where(channel_output > threshold)] = 1
                    channel_output[np.where(channel_output <= threshold)] = 0
                    plt.imsave(os.path.join(os.path.join(output_path, "mask1"),
                                            "channel_" + str(k) + ".jpeg"), channel_output)
                if j == len(layer_outputs)-1:
                    channel_output[np.where(channel_output > threshold)] = 1
                    channel_output[np.where(channel_output <= threshold)] = 0
                    plt.imsave(os.path.join(os.path.join(output_path, "mask2"),
                                            "channel_" + str(k) + ".jpeg"), channel_output)
