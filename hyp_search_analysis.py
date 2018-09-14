from imgaug import augmenters as iaa
import csv
from collections import defaultdict
import numpy as np

import os



def analyze():

    ### RANDOM SEARCH HYPER-PARAMETER OPTIONS
    tuning_constants = [.2,.5,1.0,1.5,2.0]
    ss_rs = [.166,.33,.5,.6,.667]
    objective_fns = ["wce","gdice","ss"]
    regularizer_argss = [None, None,None, None,("L1",1E-8),("L1",1E-6),("L1",1E-4),("L1",1E-2),("L2",1E-8),("L2",1E-6),
                        ("L2",1E-4),("L2",1E-2)]
    learning_rate_and_kwargss = [(.1, {"decay_epochs":5,"decay_rate":.1,"staircase":False}),
                                 (.1, {"decay_epochs":5,"decay_rate":.1,"staircase":True}),
                                 (.1, {"decay_epochs": 10, "decay_rate": .1, "staircase": False}),
                                 (.1, {"decay_epochs": 10, "decay_rate": .1, "staircase": True}),
                                 (.1, {}),
                                 (.01, {}),
                                 (.001, {})]

    op_fun_and_kwargss = [("adam", {}), ("grad", {}), ("adagrad", {}), ("adadelta", {}), ("rmsprop", {})]
    weight_inits = ["default","He","Xnormal"]
    act_fns = ["lrelu"]
    act_leak_probs = [0.0,0.0,0.2,.2,0.4,0.6]

    hist_eqs = [True,False]

    clahe_kwargss = [None, None, None, None, None,
                     {"clipLimit": 2.0,"tileGridSize":(8,8)}, {"clipLimit": 2.0,"tileGridSize":(4,4)},
                     {"clipLimit": 2.0,"tileGridSize":(16,16)}, {"clipLimit": 20.0, "tileGridSize": (8, 8)},
                     {"clipLimit": 60.0, "tileGridSize": (8, 8)}]

    per_image_normalizations = [False, True]
    gammas = [1.0,1.0,1.0,2.0,4.0,6.0]

    seqs = [None, None, iaa.Sequential([
        iaa.Crop(px=(0, 16)), # crop images from each side by 0 to 16px (randomly chosen)
        iaa.GaussianBlur(sigma=(0, 3.0)) # blur images with a sigma of 0 to 3.0
        ])]

    ### Dictionary for Job results
    auc_roc_marg_scores = defaultdict(lambda : defaultdict(lambda : defaultdict(list)))
    n_metric_intervals = 4

    list_all_hyps = ["tuning_constants","ss_rs","objective_fns","regularizer_argss","learning_rate_and_kwargss",
            "op_fun_and_kwargss","weight_inits","act_fns","act_leak_probs","hist_eqs","clahe_kwargss",
            "per_image_normalizations","gammas","seqs"]

    hyps_options=dict()
    for hyp in list_all_hyps:
        hyps_options[hyp]= locals()[hyp]

    for i in range(n_metric_intervals):
        auc_roc_marg_scores[i]

        for tuning_constant in hyps_options["tuning_constants"]:
            auc_roc_marg_scores[i]["tuning_constants"][str(tuning_constant)]

        for ss_r in hyps_options["ss_rs"]:
            auc_roc_marg_scores[i]["ss_rs"][str(ss_r)]

        for objective_fn in hyps_options["objective_fns"]:
            auc_roc_marg_scores[i]["objective_fns"][str(objective_fn)]

        for regularizer_args in hyps_options["regularizer_argss"]:
            auc_roc_marg_scores[i]["regularizer_argss"][str(regularizer_args)]

        for learning_rate_and_kwargs in hyps_options["learning_rate_and_kwargss"]:
            auc_roc_marg_scores[i]["learning_rate_and_kwargss"][str(learning_rate_and_kwargs)]

        for op_fun_and_kwargs in hyps_options["op_fun_and_kwargss"]:
            auc_roc_marg_scores[i]["op_fun_and_kwargss"][str(op_fun_and_kwargs)]

        for weight_init in hyps_options["weight_inits"]:
            auc_roc_marg_scores[i]["weight_inits"][str(weight_init)]

        for act_leak_prob in hyps_options["act_leak_probs"]:
            auc_roc_marg_scores[i]["act_leak_probs"][str(act_leak_prob)]

        for hist_eq in hyps_options["hist_eqs"]:
            auc_roc_marg_scores[i]["hist_eqs"][str(hist_eq)]

        for clahe_kwargs in hyps_options["clahe_kwargss"]:
            auc_roc_marg_scores[i]["clahe_kwargss"][str(clahe_kwargs)]

        for gamma in hyps_options["gammas"]:
            auc_roc_marg_scores[i]["gammas"][str(gamma)]

        for seq in hyps_options["seqs"]:
            if seq:
                auc_roc_marg_scores[i]["seqs"][str(True)]
            else:
                auc_roc_marg_scores[i]["seqs"][str(False)]

    filtered_hyps = [hyp for hyp in list_all_hyps if hyp in auc_roc_marg_scores[i]]

    EXPERIMENTS_DIR_PATH = "/home/ubuntu/new_vessel_segmentation/vessel-segmentation/experiments1"
    job_files = os.listdir(EXPERIMENTS_DIR_PATH)

    for job_file in job_files:
        print(job_file)
        JOB_PATH = os.path.join(EXPERIMENTS_DIR_PATH, job_file)
        job_metrics_file = [file for file in os.listdir(JOB_PATH) if "mof" in file][0]
        job_hyp_params = job_file.split(',')
        JOB_METRICS_PATH = os.path.join(JOB_PATH, job_metrics_file)

        auc_col = 1

        with open(JOB_METRICS_PATH) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            next(csv_reader, None)
            for i, row in enumerate(csv_reader):
                auc = row[auc_col]

                print(job_hyp_params)

                for job_hyp_param,hyp_name in zip(job_hyp_params,filtered_hyps):
                    hyp_options = hyps_options[hyp_name]
                    for hyp_option in hyp_options:
                        if str(hyp_option) in job_hyp_param:
                            auc_roc_marg_scores[i][hyp_name][hyp_option] = auc_roc_marg_scores[i][hyp_name][hyp_option] + [auc]

    hyp_metrics_log = "hyp_log.csv"
    hyp_metrics_log_path = os.path.join("/home/ubuntu/new_vessel_segmentation/vessel-segmentation", hyp_metrics_log)

    with open(hyp_metrics_log_path, "a") as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        hyp_keys_opts_strs = [[hyp_key+"_"+str(hyp_opt) for hyp_opt in auc_roc_marg_scores[0][hyp_key].keys()]
                              for hyp_key in filtered_hyps]
        writer.writerow(hyp_keys_opts_strs)

        for i in range(n_metric_intervals):
            metric_i_auc_roc_marg_scores = auc_roc_marg_scores[i]
            results = []
            for hyp_name in filtered_hyps:
                hyp_options = hyps_options[hyp_name]
                for hyp_option in hyp_options:
                    results += [np.mean([auc_roc_marg_scores[i][hyp_name][hyp_option]])]
            writer.writerow(results)

if __name__ == '__main__':
    analyze()














