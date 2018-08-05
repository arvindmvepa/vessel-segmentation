import multiprocessing
import os
import time

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


import datetime
from sklearn.metrics import precision_recall_fscore_support, cohen_kappa_score, roc_auc_score, confusion_matrix, \
    roc_curve, auc as auc_

from random import randint

from dataset.drive import DriveDataset
from dataset.dsa import DsaDataset
from network.base import Network
from network.drive import DriveNetwork
from network.dsa import DsaNetwork

from utilities.output_ops import tile_images
from utilities.misc import find_class_balance

# make aspects of train function part of network classes

def train(network = "drive", dataset_kwargs = {}, device = '/gpu:1',end_freq = 2000, decision_thresh = .75,
          score_freq=10, layer_output_freq=200, output_file="results.txt", cost_log="cost_log.txt", tuning_constant=1.0,
          cur_time="None", batch_size=1, n_examples=5):

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    plt.rcParams['image.cmap'] = 'gray'

    if network == "drive":
        dataset = DriveDataset(**dataset_kwargs)
    elif network == "dsa":
        dataset = DsaDataset(**dataset_kwargs)
    else:
        raise ValueError("Network Type Undefined")

    train_inputs, train_masks, train_targets = dataset.get_images_from_file()
    test_inputs, test_masks, test_targets = dataset.get_images_from_file(folder="test")

    #pos_weight calculation
    neg_pos_class_ratio, _, _ = find_class_balance(train_targets, train_masks)
    _, test_neg_class_frac, test_pos_class_frac  = find_class_balance(test_targets, test_masks)

    pos_weight = neg_pos_class_ratio * tuning_constant


    dataset.train_inputs = train_inputs
    dataset.train_masks = train_masks
    dataset.train_targets = train_targets
    dataset.test_inputs = test_inputs
    dataset.test_masks = test_masks
    dataset.test_targets = test_targets

    test_inputs = np.reshape(test_inputs, (test_inputs.shape[0], test_inputs.shape[1], test_inputs.shape[2], 1))
    test_inputs = np.multiply(test_inputs, 1.0 / 255)

    test_targets = np.reshape(test_targets, (test_targets.shape[0], test_targets.shape[1], test_targets.shape[2], 1))

    test_masks = np.reshape(test_masks, (test_masks.shape[0], test_masks.shape[1], test_masks.shape[2], 1))

    with tf.device(device):
        network = Network(wce_pos_weight=pos_weight)
    layer_count = len(network.layers)

    # create directory for saving model as well as other directories
    os.makedirs(os.path.join('save', network.description, timestamp))
    layer_output_path = os.path.join('layer_outputs', network.description, timestamp)
    os.makedirs(layer_output_path)
    layer_output_path_train = os.path.join(layer_output_path, "train")
    os.makedirs(layer_output_path_train)
    layer_output_path_test = os.path.join(layer_output_path, "test")
    os.makedirs(layer_output_path_test)

    layer_debug1_output_path_train = os.path.join(layer_output_path_train, "debug1")
    os.makedirs(layer_debug1_output_path_train)
    layer_debug2_output_path_train = os.path.join(layer_output_path_train, "debug2")
    os.makedirs(layer_debug2_output_path_train)

    layer_mask1_output_path_train = os.path.join(layer_output_path_train, "mask1")
    os.makedirs(layer_mask1_output_path_train)
    layer_mask2_output_path_train = os.path.join(layer_output_path_train, "mask2")
    os.makedirs(layer_mask2_output_path_train)

    for layer_index in range(layer_count):
        layer_output_path_train = os.path.join(layer_output_path_train, str(layer_index))
        os.makedirs(layer_output_path_train)
        layer_output_path_test = os.path.join(layer_output_path_test, str(layer_index))
        os.makedirs(layer_output_path_test)

    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        with tf.device(device):
            print(sess.run(tf.global_variables_initializer()))

            summary_writer = tf.summary.FileWriter('{}/{}-{}'.format('logs', network.description, timestamp),
                                                   graph=tf.get_default_graph())

            saver = tf.train.Saver(tf.all_variables(), max_to_keep=None)

            test_accuracies = []
            test_auc = []
            test_auc_10_fpr = []
            test_auc_05_fpr = []
            test_auc_025_fpr = []
            max_thresh_accuracies = []

            global_start = time.time()
            batch_num = 0
            n_epochs  = 4000
            for epoch_i in range(n_epochs):
                if batch_num > end_freq:
                    dataset.reset_batch_pointer()
                    break
                dataset.reset_batch_pointer()
                for batch_i in range(dataset.num_batches_in_epoch()):
                    batch_num = epoch_i * dataset.num_batches_in_epoch() + batch_i + 1
                    if batch_num > end_freq:
                        break

                    start = time.time()
                    batch_inputs, batch_masks, batch_targets = dataset.next_batch()

                    print(batch_inputs.shape)

                    plt.imsave(os.path.join(layer_output_path_train, "test1.jpeg"), batch_inputs[0])
                    plt.imsave(os.path.join(layer_output_path_train, "test1_target.jpeg"), batch_targets[0])

                    batch_inputs = np.reshape(batch_inputs, (dataset.batch_size, batch_inputs.shape[0],
                                                             batch_inputs.shape[1], 1))
                    batch_masks = np.reshape(batch_masks, (dataset.batch_size, batch_masks.shape[0],
                                                           batch_masks.shape[1], 1))
                    batch_targets = np.reshape(batch_targets, (dataset.batch_size, batch_targets.shape[0],
                                                               batch_targets.shape[1], 1))

                    plt.imsave(os.path.join(layer_output_path_train, "test2.jpeg"), batch_inputs[0, :, :, 0])
                    plt.imsave(os.path.join(layer_output_path_train, "test2_target.jpeg"), batch_targets[0, :, :, 0])

                    batch_inputs = np.multiply(batch_inputs, 1.0 / 255)

                    cost, cost_unweighted, layer_outputs, debug1, debug2, _ = sess.run([network.cost,
                                                                                        network.cost_unweighted,
                                                                           network.layer_outputs, network.debug1,
                                                                           network.debug2, network.train_op],
                                       feed_dict={network.inputs: batch_inputs, network.masks: batch_masks,
                                                  network.targets: batch_targets, network.is_training: True})
                    end = time.time()

                    debug1 = debug1[0,:,:,0]
                    debug2 = debug2[0,:,:,0]
                    plt.imsave(os.path.join(layer_debug1_output_path_train, "debug1.jpeg"), debug1)
                    plt.imsave(os.path.join(layer_debug2_output_path_train, "debug2.jpeg"), debug2)

                    channel_list = []

                    if batch_num % score_freq == 0 or batch_num == n_epochs * dataset.num_batches_in_epoch():
                        if batch_num % layer_output_freq == 0:
                            for j, layer_output in enumerate(layer_outputs):
                                for k in range(layer_output.shape[3]):
                                    channel_output = layer_output[0,:,:,k]
                                    plt.imsave(os.path.join(os.path.join(layer_output_path_train, str(j+1)),"channel_"+
                                                            str(k)+ ".jpeg"), channel_output)
                                    if j == 0:
                                        channel_output[np.where(channel_output > decision_thresh)] = 1
                                        channel_output[np.where(channel_output <= decision_thresh)] = 0
                                        plt.imsave(os.path.join(os.path.join(layer_output_path_train, "mask1"),
                                                                "channel_" + str(k) + ".jpeg"), channel_output)
                                    if j == 16:
                                        channel_output[np.where(channel_output > decision_thresh)] = 1
                                        channel_output[np.where(channel_output <= decision_thresh)] = 0
                                        plt.imsave(os.path.join(os.path.join(layer_output_path_train, "mask2"),
                                                                "channel_" + str(k) + ".jpeg"), channel_output)

                        test_accuracy = 0.0
                        max_thresh_accuracy = 0.0

                        test_cost = 0.0
                        test_cost_unweighted = 0.0

                        mask_array = np.zeros((len(test_inputs), test_masks.shape[1], test_masks.shape[2]))
                        target_array = np.zeros((len(test_inputs), test_targets.shape[1], test_targets.shape[1]))
                        prediction_array = np.zeros((len(test_inputs), test_targets.shape[1], test_targets.shape[1]))

                        sample_test_image = randint(0, len(test_inputs)-1)
                        for i, test_input in enumerate(test_inputs):
                            thresh_max = 0.0
                            if i == sample_test_image and batch_num % layer_output_freq == 0:
                                test_cost_, test_cost_unweighted_, inputs, masks, results, targets, acc, \
                                layer_outputs = sess.run(
                                    [network.cost, network.cost_unweighted, network.inputs, network.masks,
                                     network.segmentation_result, network.targets, network.accuracy,
                                     network.layer_outputs],
                                    feed_dict={network.inputs: test_input, network.masks: test_masks[i:(i + 1)],
                                               network.targets: test_targets[i:(i + 1)],network.is_training: False})

                                top_pad = int((network.FIT_IMAGE_HEIGHT - network.IMAGE_HEIGHT) / 2)
                                bot_pad = (network.FIT_IMAGE_HEIGHT - network.IMAGE_HEIGHT) - top_pad
                                left_pad = int((network.FIT_IMAGE_WIDTH - network.IMAGE_WIDTH) / 2)
                                right_pad = (network.FIT_IMAGE_WIDTH - network.IMAGE_WIDTH) - left_pad

                                target = test_targets[i:(i + 1)]
                                target = np.reshape(target, (network.IMAGE_HEIGHT, network.IMAGE_WIDTH))
                                invert_target = np.invert(target)

                                mask = test_masks[i:(i + 1)]
                                mask = np.reshape(mask, (network.IMAGE_HEIGHT, network.IMAGE_WIDTH))
                                target = np.multiply(target, mask)
                                invert_target = np.multiply(invert_target, mask)

                                channel_list = []
                                total_pos = 0
                                total_neg = 0

                                for j in range(len(layer_outputs)):
                                    layer_output = layer_outputs[j]
                                    for k in range(layer_output.shape[3]):
                                        channel_output = layer_output[0, :, :, k]
                                        plt.imsave(os.path.join(os.path.join(layer_output_path_test, str(j + 1)),
                                                              "channel_" + str(k) + ".jpeg"), channel_output)
                                        if j == 0:
                                            input = np.reshape(channel_output, (network.FIT_IMAGE_HEIGHT,
                                                                                network.FIT_IMAGE_WIDTH ))
                                            input_crop = input[top_pad:network.FIT_IMAGE_HEIGHT - bot_pad,
                                                         left_pad:network.FIT_IMAGE_WIDTH - right_pad]
                                            invert_input_crop = 1-input_crop

                                            results_ = np.multiply(input_crop, target)
                                            results_invert = np.multiply(invert_input_crop, invert_target)

                                            sum = np.sum(results_)
                                            sum_neg = np.sum(results_invert)

                                            total_pos += sum
                                            total_neg += sum_neg

                                            channel_list.append((sum, input_crop))

                                            channel_output[np.where(channel_output > decision_thresh)] = 1
                                            channel_output[np.where(channel_output <= decision_thresh)] = 0
                                            plt.imsave(os.path.join(os.path.join(layer_output_path_test, "mask1"),
                                                                    "channel_" + str(k) + ".jpeg"), channel_output)
                                        if j == 16:
                                            channel_output[np.where(channel_output > decision_thresh)] = 1
                                            channel_output[np.where(channel_output <= decision_thresh)] = 0
                                            plt.imsave(os.path.join(os.path.join(layer_output_path_test, "mask2"),
                                                                    "channel_" + str(k) + ".jpeg"),channel_output)

                            else:
                                test_cost_, test_cost_unweighted_, inputs, masks, results, targets, acc = \
                                    sess.run([network.cost, network.cost_unweighted, network.inputs, network.masks,
                                              network.segmentation_result, network.targets, network.accuracy],
                                             feed_dict={network.inputs: test_inputs[i:(i + 1)],
                                                        network.masks: test_masks[i:(i + 1)],
                                                        network.targets: test_targets[i:(i + 1)],
                                                        network.is_training: False})
                            masks = masks[0, :, :, 0]
                            results = results[0, :, :, 0]
                            targets = targets[0, :, :, 0]

                            mask_array[i] = masks
                            target_array[i] = targets
                            prediction_array[i] = results

                            test_accuracy += acc
                            test_cost += test_cost_
                            test_cost_unweighted += test_cost_unweighted_

                            fprs, tprs, thresholds = roc_curve(targets.flatten(), results.flatten(), sample_weight=masks.flatten())
                            list_fprs_tprs_thresholds = list(zip(fprs, tprs, thresholds))
                            interval = 0.0001

                            for i in np.arange(0.0, 1.0 + interval, interval):
                                index = int(round((len(thresholds) - 1) * i, 0))
                                fpr, tpr, threshold = list_fprs_tprs_thresholds[index]
                                thresh_acc = (1 - fpr) * test_neg_class_frac + tpr * test_pos_class_frac
                                if thresh_acc > thresh_max:
                                    thresh_max = thresh_acc
                                i += 1

                            max_thresh_accuracy += thresh_max


                        max_thresh_accuracy = max_thresh_accuracy / len(test_inputs)
                        test_accuracy = test_accuracy / len(test_inputs)
                        test_cost = test_cost / len(test_inputs)
                        test_cost_unweighted = test_cost_unweighted / len(test_inputs)

                        mask_flat = mask_array.flatten()
                        prediction_flat = prediction_array.flatten()
                        target_flat = target_array.flatten()

                        ## save mask and target (if files don't exist)
                        cwd = os.getcwd()
                        if not os.path.exists(os.path.join(cwd, "mask.npy")):
                            np.save("mask", mask_flat)
                        if not os.path.exists(os.path.join(cwd, "target.npy")):
                            np.save("target", target_flat)

                        ### save prediction array for ensemble
                        file_name = timestamp + "_" + str(batch_num)
                        np.save(file_name, prediction_flat)


                        auc = roc_auc_score(target_flat, prediction_flat, sample_weight=mask_flat)
                        fprs, tprs, thresholds = roc_curve(target_flat, prediction_flat, sample_weight=mask_flat)
                        np_fprs, np_tprs, np_thresholds = np.array(fprs).flatten(), np.array(tprs).flatten(), np.array(thresholds).flatten()
                        fpr_10 = np_fprs[np.where(np_fprs < .10)]
                        tpr_10 = np_tprs[0:len(fpr_10)]

                        fpr_05 = np_fprs[np.where(np_fprs < .05)]
                        tpr_05 = np_tprs[0:len(fpr_05)]

                        fpr_025 = np_fprs[np.where(np_fprs < .025)]
                        tpr_025 = np_tprs[0:len(fpr_025)]

                        #upper_thresholds = np_thresholds[0:len(fpr_10)]
                        thresh_acc_strings = ""
                        list_fprs_tprs_thresholds = list(zip(fprs, tprs, thresholds))

                        auc_10_fpr = auc_(fpr_10, tpr_10)
                        auc_05_fpr = auc_(fpr_05, tpr_05)
                        auc_025_fpr = auc_(fpr_025, tpr_025)

                        thresh_max_items = "max thresh acc : {}, ".format(max_thresh_accuracy)

                        interval = 0.05
                        for i in np.arange(0, 1.0 + interval, interval):
                            index = int(round((len(thresholds) - 1) * i, 0))
                            fpr, tpr, threshold = list_fprs_tprs_thresholds[index]
                            thresh_acc = (1 - fpr) * test_neg_class_frac + tpr * test_pos_class_frac
                            thresh_acc_strings += "thresh: {}, thresh acc: {}, tpr: {}, spec: {}, ".format(threshold, thresh_acc, tpr, 1-fpr)

                        thresh_acc_strings = thresh_max_items +thresh_acc_strings
                        result_flat = (prediction_flat > decision_thresh).astype(int)

                        prediction_flat = np.round(prediction_flat)
                        target_flat = np.round(target_flat)


                        (precision, recall, fbeta_score, _) = precision_recall_fscore_support(target_flat,
                                                                                              prediction_flat,
                                                                                              average='binary',
                                                                                              sample_weight=mask_flat)

                        kappa = cohen_kappa_score(target_flat, prediction_flat, sample_weight=mask_flat)
                        tn, fp, fn, tp = confusion_matrix(target_flat, prediction_flat, sample_weight=mask_flat).ravel()

                        specificity = tn / (tn + fp)
                        #sess.run(tf.local_variables_initializer())
                        r_tn, r_fp, r_fn, r_tp = confusion_matrix(target_flat, result_flat, sample_weight=mask_flat).ravel()

                        r_acc = (r_tp+r_tn)/(r_tp+r_tn+r_fp+r_fn)
                        r_recall = r_tp / (r_tp + r_fn)
                        r_precision = r_tp / (r_tp + r_fp)
                        r_specificity = r_tn / (r_tn + r_fp)

                        # test_accuracy1 = test_accuracy1/len(test_inputs)
                        print(
                        'Step {}, test accuracy: {}, test accuracy: {}, thresh accuracy: {}, thresh recall: {}, thresh precision: {}, thresh specificity: {}, cost_unweighted: {} recall {}, specificity {}, precision {}, fbeta_score {}, auc {}, auc_10_fpr {}, auc_05_fpr {}, auc_025_fpr {}, kappa {}, class balance {}'.format(
                            batch_num, test_accuracy, r_acc, r_recall, r_precision, r_specificity, cost, cost_unweighted,recall, specificity, precision, fbeta_score, auc, auc_10_fpr, auc_05_fpr, auc_025_fpr, kappa, neg_pos_class_ratio))

                        t_inputs, t_masks, t_targets = dataset.test_inputs.tolist()[
                                                       :n_examples], dataset.test_masks.tolist()[
                                                                     :n_examples], dataset.test_targets.tolist()[
                                                                                   :n_examples]
                        test_segmentation = []
                        for i in range(n_examples):
                            test_i = np.multiply(t_inputs[i:(i + 1)], 1.0 / 255)
                            t_mask_i = t_masks[i:(i + 1)]
                            segmentation = sess.run(network.segmentation_result, feed_dict={
                                network.inputs: np.reshape(test_i, [1, Mod_WIDTH, Mod_HEIGHT, 1]),
                                network.masks: np.reshape(t_mask_i, [1, IMAGE_WIDTH, IMAGE_HEIGHT, 1])})
                            test_segmentation.append(segmentation[0])

                        test_plot_buf = draw_results(t_inputs[:n_examples],
                                                     np.multiply(t_targets[:n_examples], 1.0 / 255), test_segmentation,
                                                     test_accuracy, network, batch_num, decision_thresh)

                        image = tf.image.decode_png(test_plot_buf.getvalue(), channels=4)
                        image = tf.expand_dims(image, 0)
                        image_summary_op = tf.summary.image("plot", image)
                        image_summary = sess.run(image_summary_op)
                        summary_writer.add_summary(image_summary)
                        f1 = open(output_file, 'a')

                        test_accuracies.append((test_accuracy, batch_num))
                        test_auc.append((auc, batch_num))
                        test_auc_10_fpr.append((auc_10_fpr, batch_num))
                        test_auc_05_fpr.append((auc_05_fpr, batch_num))
                        test_auc_025_fpr.append((auc_025_fpr, batch_num))
                        max_thresh_accuracies.append((max_thresh_accuracy, batch_num))
                        print("Accuracies in time: ", [test_accuracies[x][0] for x in range(len(test_accuracies))])
                        print(test_accuracies)
                        max_acc = max(test_accuracies)
                        max_auc = max(test_auc)
                        max_auc_10_fpr = max(test_auc_10_fpr)
                        max_auc_05_fpr = max(test_auc_05_fpr)
                        max_auc_025_fpr = max(test_auc_025_fpr)
                        max_thresh_accuracy = max(max_thresh_accuracies)
                        if batch_num % layer_output_freq == 0:
                            print("layer 1 pos-neg distribution")
                            print("total positive sum: "+str(total_pos))
                            print("total negative sum: "+str(total_neg))
                            print("percentage positive: " + str(float(total_pos)/float(total_neg+total_pos)))
                            print("percentage negative: " + str(float(total_neg)/float(total_neg+total_pos)))
                            if len(channel_list) > 0:
                                tile_images(channel_list, file_name=cur_time+"_layer1_collage.jpeg")
                        print("Best accuracy: {} in batch {}".format(max_acc[0], max_acc[1]))
                        print("Total time: {}".format(time.time() - global_start))
                        f1.write(
                            'Step {}, test accuracy: {}, thresh accuracy: {}, thresh recall: {}, thresh precision: {}, thresh specificity: {}, cost: {}, cost_unweighted: {}, recall {}, specificity {}, auc {}, auc_10_fpr {}, auc_05_fpr {}, auc_025_fpr {}, precision {}, fbeta_score {}, kappa {}, class balance {}, '
                            'max acc {} {}, max auc {} {}, max auc 10 fpr {} {}, max auc 5 fpr {} {}, max auc 2.5 fpr {} {}, sample test image {} \n'.format(
                                batch_num, test_accuracy, r_acc, r_recall, r_precision, r_specificity, cost, cost_unweighted, recall, specificity, auc, auc_10_fpr, auc_05_fpr, auc_025_fpr, precision, fbeta_score, kappa, neg_pos_class_ratio,
                                max_acc[0],max_acc[1], max_auc[0], max_auc[1], max_auc_10_fpr[0], max_auc_10_fpr[1], max_auc_05_fpr[0], max_auc_05_fpr[0], max_auc_025_fpr[0], max_auc_025_fpr[1], sample_test_image))
                        f1.write(('Step {}, '+"overall max thresh accuracy {} {}, ".format(max_thresh_accuracy[0], max_thresh_accuracy[1])+thresh_acc_strings+'\n').format(batch_num))
                        f1.close()
                        f1 = open(cost_log, 'a')
                        f1.write('Step {}, training cost {}, training cost unweighted {}, test cost {}, test cost unweighted {}\n'.format(batch_num, cost, cost_unweighted, test_cost, test_cost_unweighted))
                        f1.close()
