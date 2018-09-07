"""This is the file for the base Job class"""
import multiprocessing
import matplotlib
matplotlib.use('Agg')

import os
import time
import glob

from scipy.misc import imsave
import numpy as np
import tensorflow as tf

import datetime
from sklearn.metrics import precision_recall_fscore_support, cohen_kappa_score, roc_auc_score, confusion_matrix, \
    roc_curve, auc
from sklearn.model_selection import KFold

from random import randint

from utilities.output_ops import draw_results
import csv
from numpy import genfromtxt
from statsmodels import robust
from dataset.base import Dataset


class Job(object):

    IMAGE_PLOT_DIR = "image_plots"

    # metrics to obtain on test set
    metrics = ("test set average weighted log loss","test set average unweighted log loss",
               "training set batch weighted log loss","training set batch unweighted log loss","auc","aucfpr10",
               "aucfpr05","aucfpr025","accuracy","precision","recall","specificity","f1_score","kappa","dt accuracy",
               "dt precision","dt recall","dt specificity","dt f1_score","dt kappa","max acc from threshold")

    num_thresh_scores = 10
    decision_threshold = .75

    n_epochs = 100

    def __init__(self, OUTPUTS_DIR_PATH="."):
        if not os.path.exists(OUTPUTS_DIR_PATH):
            os.makedirs(OUTPUTS_DIR_PATH)
        self.OUTPUTS_DIR_PATH = OUTPUTS_DIR_PATH

    def run_single_model(self, **kwargs):
        # kwargs are applied to train method
        p = multiprocessing.Process(target=self.train, kwargs=kwargs)
        p.start()
        p.join()

    # only metrics are returned per fold
    # for most other output, only last is kept
    def run_cross_validation(self, n_splits=3, mof_metric="mad",**kwargs):
        # produce cv indices
        k_fold = KFold(n_splits=n_splits, shuffle=True)
        WRK_DIR_PATH = kwargs.get("WRK_DIR_PATH",".")
        TRAIN_SUBDIR = kwargs.get("TRAIN_SUBDIR","train/")
        IMAGES_DIR_PATH = os.path.join(WRK_DIR_PATH, TRAIN_SUBDIR, Dataset.IMAGES_DIR)
        imgs = os.listdir(IMAGES_DIR_PATH)

        # get base file name for logging cv results
        metrics_log = kwargs.pop("metrics_log","")
        if metrics_log == "":
            metrics_log = "metrics_log.csv"
        metrics_log_fname_lst = os.path.splitext(metrics_log)
        folds_metrics_log_fname = []

        # run job per cv fold
        for i, (train_inds, test_inds) in enumerate(k_fold.split(imgs)):
            fold_metrics_log_fname_lst = list(metrics_log_fname_lst)
            fold_suffix = "_fold_"+str(i)
            fold_metrics_log_fname_lst[0] = metrics_log_fname_lst[0] + fold_suffix
            fold_kwargs = kwargs.copy()
            fold_metrics_log_fname = "".join(fold_metrics_log_fname_lst)
            fold_kwargs["metrics_log"]= fold_metrics_log_fname
            folds_metrics_log_fname += [fold_metrics_log_fname]
            fold_kwargs["cv_train_inds"] = train_inds
            fold_kwargs["cv_test_inds"] = test_inds

            # fold_kwargs are applied to train method

            p = multiprocessing.Process(target=self.train, kwargs=fold_kwargs)
            p.start()
            p.join()

        # define func for measure of fit
        if mof_metric == "mad":
            mof_func = robust.mad
        elif mof_metric == "std":
            mof_func = np.std

        # combine results from each cv fold log
        metric_folds_results = []
        for fold_metrics_log_fname in folds_metrics_log_fname:
            fold_metrics_log_path = os.path.join(self.OUTPUTS_DIR_PATH, fold_metrics_log_fname)
            metric_fold_results = genfromtxt(fold_metrics_log_path,skip_header=1,delimiter=',')
            if len(metric_fold_results.shape) == 1:
                metric_fold_results = np.expand_dims(metric_fold_results,axis=0)
            metric_folds_results += [metric_fold_results]
        metric_folds_results = np.array(metric_folds_results)

        # calculate the mean and mof
        mean_folds_results = np.mean(metric_folds_results, axis=0)
        mof_folds_results = mof_func(metric_folds_results, axis=0)

        # create file name and path for combined results
        combined_metrics_log_fname_lst = list(metrics_log_fname_lst)
        combined_metrics_log_fname_lst[0] = combined_metrics_log_fname_lst[0] + "_combined_mof_"+mof_metric
        combined_metrics_log_fname = "".join(combined_metrics_log_fname_lst)
        combined_metrics_log_path = os.path.join(self.OUTPUTS_DIR_PATH, combined_metrics_log_fname)

        num_thresh_scores = kwargs.get("num_thresh_scores", self.num_thresh_scores)

        # create results file with combined results
        self.write_to_csv(sorted(self.get_metric_names(num_thresh_scores)), combined_metrics_log_path)
        for i in range(len(mean_folds_results)):
            row = [" +/- ".join([str(entry[0]),str(entry[1])]) for entry in zip(mean_folds_results[i],
                                                                                mof_folds_results[i])]
            self.write_to_csv(row, combined_metrics_log_path)

    def run_ensemble(self, ensemble_count=10, combining_metric="mean", wce_start_tuning_constant=.5,
                     wce_end_tuning_constant=2.0, wce_tuning_constants=None, **kwargs):

        if wce_tuning_constants is not None:
            assert len(wce_tuning_constants) == ensemble_count
        elif wce_start_tuning_constant is not None and wce_end_tuning_constant is not None:
            if ensemble_count == 1:
                wce_tuning_constants = [wce_start_tuning_constant]
            # uniformly distribute tuning constants used for ensemble if wce_tuning_constants not provided
            else:
                interval = (wce_end_tuning_constant - wce_start_tuning_constant) / float(ensemble_count - 1)
                wce_tuning_constants = list(
                    np.arange(wce_start_tuning_constant, wce_end_tuning_constant + interval, interval))
        else:
            wce_tuning_constant = wce_start_tuning_constant

        metrics_log = kwargs.pop("metrics_log","")
        if metrics_log == "":
            metrics_log = "ensemble_metrics_log.csv"
        metrics_log_fname_lst = os.path.splitext(metrics_log)

        # kwargs are applied to dataset class to obtain `test_neg_class_frac` and `test_pos_class_frac`
        dataset = self.dataset_cls(**kwargs)
        kwargs["dataset"] = dataset

        for i in range(ensemble_count):
            model_kwargs = kwargs.copy()
            model_metrics_log_fname_lst = list(metrics_log_fname_lst)
            model_suffix = "_model_"+str(i)
            model_metrics_log_fname_lst[0] = metrics_log_fname_lst[0] + model_suffix
            model_metrics_log_fname = "".join(model_metrics_log_fname_lst)
            model_kwargs["metrics_log"] = model_metrics_log_fname

            # select from tuning constants if provided
            if wce_tuning_constants is not None:
                wce_tuning_constant = wce_tuning_constants[i]

            model_kwargs["tuning_constant"] = wce_tuning_constant

            p = multiprocessing.Process(target=self.train, kwargs=model_kwargs)
            p.start()
            p.join()


        n_epochs = kwargs.pop("n_epochs", self.n_epochs)
        num_thresh_scores = kwargs.pop("num_thresh_scores", self.num_thresh_scores)
        decision_threshold = kwargs.pop("decision_threshold", self.decision_threshold)

        # obtain `test_neg_class_frac` and `test_pos_class_frac`
        _, test_neg_class_frac, test_pos_class_frac = dataset.get_inverse_pos_freq(*dataset.test_data[1:])

        # add metric names to ensemble metrics log
        self.write_to_csv(sorted(self.get_metric_names(num_thresh_scores)), os.path.join(self.OUTPUTS_DIR_PATH,
                                                                                         metrics_log))

        self.get_ensemble_results(n_epochs, decision_threshold, num_thresh_scores, test_neg_class_frac,
                                  test_pos_class_frac, metrics_log=metrics_log, combining_metric=combining_metric,
                                  **kwargs)

    
    def train(self, dataset=None, gpu_device=None, tuning_constant=1.0, metrics_epoch_freq=1, viz_layer_epoch_freq=10,
              metrics_log="metrics_log.csv", num_image_plots=5, save_model=True, debug_net_output=True,
              objective_fn="wce", weight_map=None, type_weight='Custom', ss_r=.05, regularizer_args=None,
              learning_rate_and_kwargs=(.001, {}), op_fun_and_kwargs=("adam", {}), weight_init=None, act_fn="lrelu",
              layer_params=None, **kwargs):

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")

        # get `n_epochs` values from kwargs if exists
        if "n_epochs" in kwargs:
            self.n_epochs = kwargs.pop("n_epochs")
        # get `num_thresh_scores` values from kwargs if exists
        if "num_thresh_scores" in kwargs:
            self.num_thresh_scores = kwargs.pop("num_thresh_scores")
        # get `decision_threshold` values from kwargs if exists
        if "decision_threshold" in kwargs:
            self.decision_threshold = kwargs.pop("decision_threshold")

        # kwargs are applied to dataset class
        if dataset is None:
            dataset = self.dataset_cls(**kwargs)

        pos_weight = dataset.get_tuned_pos_ce_weight(tuning_constant, *dataset.train_data[1:])

        # initialize network object
        if gpu_device is not None:
            with tf.device(gpu_device):
                network = self.network_cls(pos_weight=pos_weight, objective_fn=objective_fn, weight_map=weight_map,
                                           type_weight=type_weight, r=ss_r, weight_init=weight_init,
                                           regularizer_args=regularizer_args, act_fn=act_fn,
                                           learning_rate_and_kwargs=learning_rate_and_kwargs,
                                           op_fun_and_kwargs=op_fun_and_kwargs, layer_params=layer_params,
                                           num_batches_in_epoch = dataset.num_batches_in_epoch())
        else:
            network = self.network_cls(pos_weight=pos_weight, objective_fn=objective_fn, weight_map=weight_map,
                                       type_weight=type_weight, r=ss_r, weight_init=weight_init,
                                       regularizer_args=regularizer_args, act_fn=act_fn,
                                       learning_rate_and_kwargs=learning_rate_and_kwargs,
                                       op_fun_and_kwargs=op_fun_and_kwargs, layer_params=layer_params,
                                       num_batches_in_epoch=dataset.num_batches_in_epoch())

        # create metrics log file
        metric_log_file_path = os.path.join(self.OUTPUTS_DIR_PATH, metrics_log)
        self.write_to_csv(sorted(self.get_metric_names(self.num_thresh_scores)), metric_log_file_path)

        # create summary writer
        summary_writer = tf.summary.FileWriter(
            '{}/{}/{}-{}'.format(self.OUTPUTS_DIR_PATH, 'logs', network.description,
                                 timestamp),
            graph=tf.get_default_graph())

        # create directories and subdirectories
        if save_model:
            os.makedirs(os.path.join(self.OUTPUTS_DIR_PATH, 'save', network.description, timestamp))

        if viz_layer_epoch_freq is not None:
            viz_layer_outputs_path_train, viz_layer_outputs_path_test = \
                self.create_viz_dirs(network, timestamp)

        config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            tf.train.Saver(tf.all_variables(), max_to_keep=None)

            # loop over epochs
            for epoch_i in range(self.n_epochs):
                dataset.reset_batch_pointer()
                # loop over batches in epoch
                for batch_i in range(dataset.num_batches_in_epoch()):
                    start = time.time()
                    batch_num = epoch_i * dataset.num_batches_in_epoch() + batch_i

                    batch_data = dataset.next_batch()

                    # produce debug image 1
                    if viz_layer_epoch_freq is not None and debug_net_output:
                        self.save_debug1(batch_data, viz_layer_outputs_path_train)

                    # reshape array for tensorflow
                    batch_data = dataset.tf_reshape(batch_data)

                    # produce debug image 2
                    if viz_layer_epoch_freq is not None and debug_net_output:
                        self.save_debug2(batch_data, viz_layer_outputs_path_train)

                    # train on batch
                    cost, cost_unweighted, layer_outputs, debug1, _, cur_learning_rate = sess.run(
                        [network.cost, network.cost_unweighted, network.layer_outputs, network.debug1,
                         network.train_op, network.cur_learning_rate],
                        feed_dict=self.get_network_dict(network, batch_data))
                    end = time.time()
                    # print training information
                    print(network.objective_fn)
                    print('{}/{}, epoch: {}, cost: {}, cost unweighted: {}, batch time: {}, positive_weight: {}, learning rate {}'.format(
                        batch_num, self.n_epochs * dataset.num_batches_in_epoch(), epoch_i, cost, cost_unweighted, end-start, pos_weight, cur_learning_rate))

                    # produce debug image 3
                    if viz_layer_epoch_freq is not None and debug_net_output:
                        self.save_debug3(batch_data,debug1,viz_layer_outputs_path_train)

                    # create network visualization output
                    if (epoch_i + 1) % viz_layer_epoch_freq == 0 and batch_i == dataset.num_batches_in_epoch()-1:
                        self.create_viz_layer_output(layer_outputs, self.decision_threshold,
                                                     viz_layer_outputs_path_train)

                    # calculate results on test set
                    if (epoch_i + 1) % metrics_epoch_freq == 0 and batch_i == dataset.num_batches_in_epoch()-1:
                        self.get_results_on_test_set(metric_log_file_path, network, dataset, sess,
                                                     self.decision_threshold, epoch_i, timestamp, viz_layer_epoch_freq,
                                                     viz_layer_outputs_path_test, num_image_plots, summary_writer,
                                                     cost=cost, cost_unweighted=cost_unweighted)

    def get_results_on_test_set(self, metrics_log_file_path, network, dataset, sess, decision_threshold, epoch_i,
                                timestamp, viz_layer_epoch_freq, viz_layer_outputs_path_test, num_image_plots,
                                summary_writer, **kwargs):

        max_thresh_accuracy = 0.0
        test_cost = 0.0
        test_cost_unweighted = 0.0

        segmentation_results = np.zeros((dataset.test_targets.shape[0], dataset.test_targets.shape[1],
                                         dataset.test_targets.shape[2]))
        sample_test_image = randint(0, len(dataset.test_images) - 1)
        # get test results per image
        for i, test_data in enumerate(zip(*dataset.test_data)):
            test_data = dataset.tf_reshape(test_data)
            # get network results on test image
            test_cost_, test_cost_unweighted_, segmentation_test_result, layer_outputs = \
                sess.run([network.cost, network.cost_unweighted, network.segmentation_result,
                          network.layer_outputs],
                         feed_dict=self.get_network_dict(network, test_data, False))

            segmentation_test_result = segmentation_test_result[0, :, :, 0]
            segmentation_results[i, :, :] = segmentation_test_result

            test_cost += test_cost_
            test_cost_unweighted += test_cost_unweighted_

            _, test_neg_class_frac, test_pos_class_frac = dataset.get_inverse_pos_freq(*test_data[1:])

            # calculate max threshold accuracy per test image
            thresh_max = self.get_max_threshold_accuracy_image(segmentation_test_result, test_neg_class_frac,
                                                               test_pos_class_frac, *test_data[1:])
            max_thresh_accuracy += thresh_max

            if i == sample_test_image and (epoch_i + 1) % viz_layer_epoch_freq == 0:
                self.create_viz_layer_output(layer_outputs, decision_threshold, viz_layer_outputs_path_test)

        # combine test results to produce overall metric scores
        max_thresh_accuracy = max_thresh_accuracy / len(dataset.test_images)
        test_cost = test_cost / len(dataset.test_images)
        test_cost_unweighted = test_cost_unweighted / len(dataset.test_images)

        prediction_flat = segmentation_results.flatten()
        target_flat = np.round(dataset.test_targets.flatten())
        mask_flat = self.get_test_mask_flat(dataset)

        # save target (if files don't exist)
        self.save_data(prediction_flat, target_flat, mask_flat, timestamp, epoch_i)

        # get class proportion on test set
        _, test_neg_class_frac, test_pos_class_frac = dataset.get_inverse_pos_freq(*dataset.test_data[1:])

        acc = self.get_metrics_on_test_set(metrics_log_file_path, prediction_flat, target_flat, mask_flat,
                                           decision_threshold, self.num_thresh_scores, test_neg_class_frac,
                                           test_pos_class_frac, max_thresh_accuracy=max_thresh_accuracy,
                                           cost=kwargs['cost'], cost_unweighted=kwargs['cost_unweighted'],
                                           test_cost=test_cost, test_cost_unweighted=test_cost_unweighted)

        # produce image plots
        test_plot_buf = draw_results(dataset.test_images[:num_image_plots],
                                     dataset.test_targets[:num_image_plots],
                                     segmentation_results[:num_image_plots, :, :],
                                     acc, network, epoch_i, num_image_plots, os.path.join(self.OUTPUTS_DIR_PATH,
                                                                                          self.IMAGE_PLOT_DIR),
                                     decision_threshold)

        image = tf.image.decode_png(test_plot_buf.getvalue(), channels=4)
        image = tf.expand_dims(image, 0)
        image_summary_op = tf.summary.image("plot", image)
        image_summary = sess.run(image_summary_op)
        summary_writer.add_summary(image_summary)

    @classmethod
    def get_metrics_on_test_set(cls, metrics_log_file_path, prediction_flat, target_flat, mask_flat, decision_threshold,
                                num_thresh_scores, test_neg_class_frac, test_pos_class_frac, **kwargs):

        metric_scores = dict()

        # produce AUCROC score with map
        auc_score = roc_auc_score(target_flat, prediction_flat, sample_weight=mask_flat)

        # produce auc_score curve thresholded at different FP poinnts
        fprs, tprs, thresholds = roc_curve(target_flat, prediction_flat, sample_weight=mask_flat)
        np_fprs, np_tprs, np_thresholds = np.array(fprs).flatten(), np.array(tprs).flatten(), \
                                          np.array(thresholds).flatten()

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


        interval = 1.0 / num_thresh_scores

        threshold_scores = []
        for i in np.arange(0, 1.0 + interval, interval):
            index = int(round((len(list_fprs_tprs_thresholds) - 1) * i, 0))
            fpr, tpr, threshold = list_fprs_tprs_thresholds[index]
            thresh_acc = (1 - fpr) * test_neg_class_frac + tpr * test_pos_class_frac
            threshold_scores.append((threshold, thresh_acc, tpr, 1 - fpr))

        # produce metrics based on predictions given by decisions thresholded at .5
        rounded_prediction_flat = np.round(prediction_flat)
        (precision, recall, fbeta_score, _) = precision_recall_fscore_support(target_flat,
                                                                              rounded_prediction_flat,
                                                                              average='binary',
                                                                              sample_weight=mask_flat)

        tn, fp, fn, tp = confusion_matrix(target_flat, rounded_prediction_flat).ravel()
        kappa = cohen_kappa_score(target_flat, rounded_prediction_flat)
        acc = float(tp + tn) / float(tp + tn + fp + fn)
        specificity = float(tn) / float(tn + fp)

        # produce metrics based on predictions given by decision_threshold
        thresh_prediction_flat = (prediction_flat > decision_threshold).astype(int)

        (r_precision, r_recall, r_fbeta_score, _) = precision_recall_fscore_support(target_flat,
                                                                                    thresh_prediction_flat,
                                                                                    average='binary',
                                                                                    sample_weight=mask_flat)

        r_tn, r_fp, r_fn, r_tp = confusion_matrix(target_flat, thresh_prediction_flat, sample_weight=mask_flat).ravel()
        r_kappa = cohen_kappa_score(target_flat, thresh_prediction_flat, sample_weight=mask_flat)
        r_acc = float(r_tp + r_tn) / float(r_tp + r_tn + r_fp + r_fn)
        r_specificity = float(r_tn) / float(r_tn + r_fp)

        metric_scores["test set average weighted log loss"] = kwargs["test_cost"]
        metric_scores["test set average unweighted log loss"] = kwargs["test_cost_unweighted"]
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

        metric_scores["max acc from threshold"] = kwargs["max_thresh_accuracy"]

        for threshold_score, threshold_str in \
                zip(threshold_scores, cls.get_thresh_scores_strs(num_thresh_scores)):
            metric_scores[threshold_str[0]] = threshold_score[0]
            metric_scores[threshold_str[1]] = threshold_score[1]
            metric_scores[threshold_str[2]] = threshold_score[2]
            metric_scores[threshold_str[3]] = threshold_score[3]

        # save metric results to log
        cls.write_to_csv([metric_scores[key] for key in sorted(metric_scores.keys())], metrics_log_file_path, **kwargs)

        return acc

    def get_ensemble_results(self, n_epochs, decision_threshold, num_thresh_scores, test_neg_class_frac, test_pos_class_frac,
                             metrics_log='ensemble_results.txt', combining_metric="mean", **kwargs):

        metric_log_file_path = os.path.join(self.OUTPUTS_DIR_PATH, metrics_log)

        ## load targets and masks once for test set
        targets_path = os.path.join(self.OUTPUTS_DIR_PATH, "saved_targets", "target.npy")
        masks_path = os.path.join(self.OUTPUTS_DIR_PATH, "saved_masks", "mask.npy")
        preds_dir_path = os.path.join(self.OUTPUTS_DIR_PATH, "saved_preds")

        mask_flat = np.load(masks_path) if os.path.exists(masks_path) else None
        target_flat = np.load(targets_path)

        ## for all iterations
        for epoch_i in range(n_epochs):

            suffix = "_" + str(epoch_i) + ".npy"
            ## load all the files for an iteration
            search_string = preds_dir_path +"/*" + suffix
            model_results_files = glob.glob(search_string)

            net_results_list = []

            print(model_results_files)
            if len(model_results_files) == 0:
                continue

            for result_file in model_results_files:
                prediction_flat = np.load(result_file)
                net_results_list += [prediction_flat]

            if combining_metric == "mean":
                prediction_flat = np.array(net_results_list).mean(0)
            if combining_metric == "median":
                prediction_flat = np.median(np.array(net_results_list),0)

            self.get_metrics_on_test_set(metric_log_file_path, prediction_flat,target_flat,mask_flat, decision_threshold,
                                         num_thresh_scores, test_neg_class_frac, test_pos_class_frac,
                                         max_thresh_accuracy=np.nan, cost=np.nan, cost_unweighted=np.nan,
                                         test_cost=np.nan, test_cost_unweighted=np.nan, **kwargs)

    def create_viz_dirs(self, network, timestamp):
        viz_layer_outputs_path = os.path.join(self.OUTPUTS_DIR_PATH, 'viz_layer_outputs', network.description, timestamp)
        os.makedirs(viz_layer_outputs_path)
        viz_layer_outputs_path_train = os.path.join(viz_layer_outputs_path, "train")
        os.makedirs(viz_layer_outputs_path_train)
        viz_layer_outputs_path_test = os.path.join(viz_layer_outputs_path, "test")
        os.makedirs(viz_layer_outputs_path_test)
        # number of layer directories doubled to account for layers in reverse direction
        for i in range(2*len(network.layers)):
            viz_layer_output_path_train = os.path.join(viz_layer_outputs_path_train, str(i))
            os.makedirs(viz_layer_output_path_train)
            viz_layer_output_path_test = os.path.join(viz_layer_outputs_path_test, str(i))
            os.makedirs(viz_layer_output_path_test)
        viz_layer_mask1_output_path_train = os.path.join(viz_layer_outputs_path_train, "mask1")
        os.makedirs(viz_layer_mask1_output_path_train)
        viz_layer_mask2_output_path_train = os.path.join(viz_layer_outputs_path_train, "mask2")
        os.makedirs(viz_layer_mask2_output_path_train)
        viz_layer_mask1_output_path_test = os.path.join(viz_layer_outputs_path_test, "mask1")
        os.makedirs(viz_layer_mask1_output_path_test)
        viz_layer_mask2_output_path_test = os.path.join(viz_layer_outputs_path_test, "mask2")
        os.makedirs(viz_layer_mask2_output_path_test)
        return viz_layer_outputs_path_train, viz_layer_outputs_path_test

    def create_viz_layer_output(self, layer_outputs, threshold, output_path):
        thresh_pix_val = np.round(threshold*255)
        for j, layer_output in enumerate(layer_outputs):
            for k in range(layer_output.shape[3]):
                channel_output = np.array(np.round(layer_output[0,:,:,k]*255), dtype=np.uint8)
                imsave(os.path.join(os.path.join(output_path, str(j)),"channel_"+str(k)+".jpeg"),channel_output)
                if j == 0:
                    channel_output[np.where(channel_output > thresh_pix_val)] = 255
                    channel_output[np.where(channel_output <= thresh_pix_val)] = 0
                    imsave(os.path.join(os.path.join(output_path,"mask1"),"channel_"+str(k)+".jpeg"),channel_output)
                if j == len(layer_outputs) - 1:
                    channel_output[np.where(channel_output > thresh_pix_val)] = 255
                    channel_output[np.where(channel_output <= thresh_pix_val)] = 0
                    imsave(os.path.join(os.path.join(output_path, "mask2"),"channel_"+str(k)+".jpeg"),channel_output)

    @staticmethod
    def write_to_csv(entries, file_path, **kwargs):
        with open(file_path, "a") as csv_file:
            writer = csv.writer(csv_file, delimiter=',')
            writer.writerow(entries)

    def save_data(self, prediction_flat, target_flat, mask_flat, timestamp, epoch_i):
        targets_path = os.path.join(self.OUTPUTS_DIR_PATH, "saved_targets")
        preds_path = os.path.join(self.OUTPUTS_DIR_PATH, "saved_preds")
        if not os.path.exists(targets_path):
            os.makedirs(targets_path)
        if not os.path.exists(preds_path):
            os.makedirs(preds_path)
        if not os.path.exists(os.path.join(targets_path, "target.npy")):
            np.save(os.path.join(targets_path,"target.npy"), target_flat)

        # save prediction array e.g. for ensemble processing
        file_name = timestamp + "_" + str(epoch_i)
        np.save(os.path.join(preds_path,file_name), prediction_flat)

    @staticmethod
    def save_debug1(input_data, save_path):
        test1 = np.array(np.round(input_data[0][0,:,:]*255), dtype=np.uint8)
        test1_target = np.array(np.round(input_data[len(input_data)-1][0,:,:]*255), dtype=np.uint8)
        imsave(os.path.join(save_path, "test1.jpeg"), test1)
        imsave(os.path.join(save_path, "test1_target.jpeg"), test1_target)

    @staticmethod
    def save_debug2(input_data, save_path):
        test2 = np.array(np.round(input_data[0][0,:,:,0]*255), dtype=np.uint8)
        test2_target = np.array(np.round(input_data[len(input_data)-1][0,:,:,0]*255), dtype=np.uint8)
        imsave(os.path.join(save_path, "test2.jpeg"), test2)
        imsave(os.path.join(save_path, "test2_target.jpeg"), test2_target)

    @staticmethod
    def save_debug3(input_data, debug_data, save_path):
        test3 = np.array(np.round(input_data[0][0,:,:,0]*255), dtype=np.uint8)
        test3_target = np.array(np.round(input_data[len(input_data)-1][0,:,:,0]*255), dtype=np.uint8)
        imsave(os.path.join(save_path, "test2.jpeg"), test3)
        imsave(os.path.join(save_path, "test2_target.jpeg"), test3_target)
        debug1 = debug_data[0,:,:,0]
        debug1 = np.array(np.round(debug1*255), dtype=np.uint8)
        imsave(os.path.join(save_path, "debug1.jpeg"), debug1)

    @staticmethod
    def get_metric_names(num_thresh_scores):
        return list(Job.metrics)+sum(Job.get_thresh_scores_strs(num_thresh_scores),[])

    @staticmethod
    def get_thresh_scores_strs(num_thresh_scores):
        thresh_strs = []
        for i in range(num_thresh_scores):
            thresh_str_base = "threshold scores" + str(i)
            thresh_str_thresh = thresh_str_base + " threshold"
            thresh_str_acc = thresh_str_base + " acc"
            thresh_str_recall = thresh_str_base + " recall"
            thresh_str_spec = thresh_str_base + " spec"
            thresh_strs += [[thresh_str_thresh,thresh_str_acc,thresh_str_recall,thresh_str_spec]]
        return thresh_strs


    @property
    def dataset_cls(self):
        raise ValueError("Property Not Defined")

    @property
    def network_cls(self):
        raise ValueError("Property Not Defined")

    @staticmethod
    def get_max_threshold_accuracy_image(results, neg_class_frac, pos_class_frac, *args):
        raise NotImplementedError("Not Implemented")


    def get_test_mask_flat(self, dataset):
        return None

    def get_network_dict(self, network, input_data, train=True):
        if train:
            return {network.is_training: True}
        else:
            return {network.is_training: False}

    def __call__(self):
        pass
