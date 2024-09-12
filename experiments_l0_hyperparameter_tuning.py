import sys
import os
import tempfile
import argparse
import logging
from functools import partial
import numpy as np
import torch
import time
import secrets
import multiprocessing

import statistics

import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

from ray import train, tune
from ray.train import RunConfig
from ray.air.integrations.wandb import WandbLoggerCallback, setup_wandb

from mllp.utils import read_csv, DBEncoder
from mllp.models_l0 import L0MLLP

DATA_DIR = 'dataset'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def plot_loss(args, loss_log, accuracy, accuracy_b, f1_score, f1_score_b):
    set_name = 'validation' if args.use_validation_set else 'training'

    fig = plt.figure(figsize=(16, 16))
    fig.suptitle('Dataset: {}'.format(args.data_set), fontsize=16)
    plt.subplot(3, 1, 1)
    loss_array = np.array(loss_log)

    plt.plot(loss_array, color='b', label='Total loss')

    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('Loss during the training')
    plt.grid(True)
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(np.array(accuracy), color='b', label='MLLP')
    plt.plot(np.array(accuracy_b), color='g', label='CRS')

    plt.xlabel('epoch * 5')
    plt.ylabel('Accuracy')
    plt.title('Accuracy on the {} set'.format(set_name))
    plt.grid(True)
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.plot(np.array(f1_score), color='b', label='MLLP')
    plt.plot(np.array(f1_score_b), color='g', label='CRS')

    plt.xlabel('epoch * 5')
    plt.ylabel('F1 Score Micro')
    plt.title('F1 Score (Macro) on the {} set'.format(set_name))
    plt.grid(True)
    plt.legend()

    plt.savefig(args.plot_file)
  
def experiment(args, data_path, info_path):

    args = argparse.Namespace(**args) # included to convert back the dict required for the tuner to the args object

    print(args)

    wandb = setup_wandb(vars(args), rank_zero_only=False, entity='mllp_l0', project=args.project_name, name=f"experiment_{secrets.token_hex(3)}" if not args.hyperparameter_tuning else None)

    # Create temp dir
    tempdirname = tempfile.TemporaryDirectory().name

    args.folder_name = "artifacts"

    # args.folder_name = 'l0_{}_k{}_ki{}_useValidationSet{}_e{}_bs{}_lr{}_lrdr{}_lrde{}_wd{}_useNOT{}_lamba{}_droprate_init_input{}_droprate_init{}_N{}_beta_ema{}_local_rep{}_temperature{}_group_l0{}'.format(
    #     args.data_set, args.kfold, args.ith_kfold, args.use_validation_set, args.epoch, args.batch_size,
    #     args.learning_rate, args.lr_decay_rate, args.lr_decay_epoch, args.weight_decay, args.use_not, args.lamba, args.droprate_init_input, args.droprate_init, args.N, args.beta_ema, args.local_rep, args.temperature, args.group_l0)

    if not os.path.exists(os.path.join(tempdirname, 'log_folder')):
        os.makedirs(os.path.join(tempdirname, 'log_folder'))
    # args.folder_name = args.folder_name + '_L' + args.structure
    args.folder_path = os.path.join(tempdirname, 'log_folder', args.folder_name)
    if not os.path.exists(args.folder_path):
        os.makedirs(args.folder_path)
    # args.model = os.path.join(args.folder_path, 'model.pth')
    # args.crs_file = os.path.join(args.folder_path, 'crs.txt')
    # args.plot_file = os.path.join(args.folder_path, 'plot_file.pdf')
    args.log = os.path.join(args.folder_path, 'log.txt')

    logging.basicConfig(level=logging.INFO, filename=args.log, filemode='w', format='[%(levelname)s] - %(message)s')
    

    # dataset = args.data_set

    # data_path = os.path.join(DATA_DIR, dataset + '.data')
    # info_path = os.path.join(DATA_DIR, dataset + '.info')

    X_df, y_df, f_df, label_pos = read_csv(data_path, info_path, shuffle=True)

    kf = KFold(n_splits=args.kfold, shuffle=True, random_state=0)

    loss_ikf = []
    accuracy_ikf = []
    accuracy_b_ikf = []
    f1_score_ikf = []
    f1_score_b_ikf = []
    accuracy_v_ikf = []
    accuracy_v_b_ikf = []
    f1_score_v_ikf = []
    f1_score_v_b_ikf = []
    test_accuracy_ikf = []
    test_accuracy_b_ikf = []
    test_f1_score_ikf = []
    test_f1_score_b_ikf = []
    total_mask_active_weights_list_ikf = []
    total_active_weights_list_ikf = []
    total_mask_fully_active_weights_list_ikf = []
    total_rules_sizes_ikf = []
    k_run_time = []

    for ikf in range(args.kfold):
        start = time.time()

        # Define ikf file names
        args.model = os.path.join(args.folder_path, 'model_{ikf}.pth'.format(ikf=ikf))
        args.crs_file = os.path.join(args.folder_path, 'crs_{ikf}.txt'.format(ikf=ikf))
        args.plot_file = os.path.join(args.folder_path, 'plot_file_{ikf}.pdf'.format(ikf=ikf))
        # args.log = os.path.join(args.folder_path, 'log_{ikf}.txt')

        # logging.basicConfig(level=logging.INFO, filename=args.log, filemode='w', format='[%(levelname)s] - %(message)s')

        train_index, test_index = list(kf.split(X_df))[ikf]

        X_train_df = X_df.iloc[train_index]
        y_train_df = y_df.iloc[train_index]
        X_test_df = X_df.iloc[test_index]
        y_test_df = y_df.iloc[test_index]

        logging.info('Discretizing and binarizing data. Please wait ...')
        db_enc = DBEncoder(f_df, discrete=True)
        db_enc.fit(X_df, y_df)
        X_fname = db_enc.X_fname
        y_fname = db_enc.y_fname
        X_train, y_train = db_enc.transform(X_train_df, y_train_df)
        X_test, y_test = db_enc.transform(X_test_df, y_test_df)
        logging.info('Data discretization and binarization are done.')

        if args.use_validation_set:
            # Use 20% of the training set as the validation set.
            # kf = KFold(n_splits=5, shuffle=True, random_state=0) # In the original implementation this is hard-coded and could be missaligned with the above kfold split?
            kf = KFold(n_splits=args.kfold, shuffle=True, random_state=0)
            train_index, validation_index = next(kf.split(X_train))
            X_validation = X_train[validation_index]
            y_validation = y_train[validation_index]
            X_train = X_train[train_index]
            y_train = y_train[train_index]
        else:
            X_validation = None
            y_validation = None

        net_structure = [len(X_fname)] + list(map(int, args.structure.split('_'))) + [len(y_fname)]
        net = L0MLLP(net_structure,
                device=device,
                random_binarization_rate=args.random_binarization_rate,
                use_not=args.use_not,
                log_file=None,
                N=args.N if args.N is not None else len(X_train_df),
                beta_ema=args.beta_ema,
                weight_decay=args.weight_decay,
                lamba=args.lamba,
                droprate_init_input=args.droprate_init_input,
                droprate_init=args.droprate_init,
                temperature=args.temperature,
                group_l0=args.group_l0,
                use_bias=args.use_bias)
        net.to(device)

        loss_log, accuracy, accuracy_b, f1_score, f1_score_b, accuracy_v, accuracy_v_b, f1_score_v, f1_score_v_b, total_mask_active_weights_list, total_active_weights_list, total_mask_fully_active_weights_list, total_weights = net.train_model(
            X_train,
            y_train,
            X_validation=X_validation,
            y_validation=y_validation,
            lr=args.learning_rate,
            batch_size=args.batch_size,
            epoch=args.epoch,
            lr_decay_rate=args.lr_decay_rate,
            lr_decay_epoch=args.lr_decay_epoch)
        loss_ikf.append(loss_log)
        accuracy_ikf.append(accuracy)
        accuracy_b_ikf.append(accuracy_b)
        f1_score_ikf.append(f1_score)
        f1_score_b_ikf.append(f1_score_b)
        accuracy_v_ikf.append(accuracy_v)
        accuracy_v_b_ikf.append(accuracy_v_b)
        f1_score_v_ikf.append(f1_score_v)
        f1_score_v_b_ikf.append(f1_score_v_b)
        total_mask_active_weights_list_ikf.append(total_mask_active_weights_list)
        total_active_weights_list_ikf.append(total_active_weights_list)
        total_mask_fully_active_weights_list_ikf.append(total_mask_fully_active_weights_list)

        plot_loss(args, loss_log, accuracy, accuracy_b, f1_score, f1_score_b)

        acc, acc_b, f1, f1_b = net.test(X_test, y_test, need_transform=True)
        logging.info('=' * 60)
        logging.info('Test:\n\tAccuracy of MLLP Model: {}\n\tAccuracy of CRS  Model: {}'.format(acc, acc_b))
        logging.info('Test:\n\tF1 Score of MLLP Model: {}\n\tF1 Score of CRS  Model: {}'.format(f1, f1_b))
        logging.info('=' * 60)

        test_accuracy_ikf.append(acc)
        test_accuracy_b_ikf.append(acc_b)
        test_f1_score_ikf.append(f1)
        test_f1_score_b_ikf.append(f1_b)

        with open(args.crs_file, 'w') as f:
            rules_list = net.concept_rule_set_print(X_train, X_fname, y_fname, f)
        torch.save(net.state_dict(), args.model)

        rules_sizes = []

        for i, layer_rules in enumerate(rules_list):
            layer_rules_sizes = []
            for rule in layer_rules:
                layer_rules_sizes.append(len(layer_rules[rule]))
            rules_sizes.append(layer_rules_sizes)

        total_rules_sizes = 0
        for layer_rules_sizes in rules_sizes:
            total_rules_sizes += sum(layer_rules_sizes)

        total_rules_sizes_ikf.append(total_rules_sizes)

        end = time.time()

        k_run_time.append(end - start)
    
        logging.info(f"Total execution time: {end - start} seconds.")

    # Send the logs to Tune and Wandb
    for i in range(0, args.epoch):
        report_dict = {"epoch": i}
        report_dict.update({"loss_avg": statistics.mean([loss_ikf[ikf][i] for ikf in range(args.kfold)])})
        report_dict.update({"loss_sd": statistics.stdev([loss_ikf[ikf][i] for ikf in range(args.kfold)])})
        report_dict.update({"total_mask_active_weights_avg": statistics.mean([total_mask_active_weights_list_ikf[ikf][i] for ikf in range(args.kfold)])})
        report_dict.update({"total_mask_active_weights_sd": statistics.stdev([total_mask_active_weights_list_ikf[ikf][i] for ikf in range(args.kfold)])})
        report_dict.update({"total_active_weights_avg": statistics.mean([total_active_weights_list_ikf[ikf][i] for ikf in range(args.kfold)])})
        report_dict.update({"total_active_weights_sd": statistics.stdev([total_active_weights_list_ikf[ikf][i] for ikf in range(args.kfold)])})
        report_dict.update({"total_mask_fully_active_weights_avg": statistics.mean([total_mask_fully_active_weights_list_ikf[ikf][i] for ikf in range(args.kfold)])})
        report_dict.update({"total_mask_fully_active_weights_sd": statistics.stdev([total_mask_fully_active_weights_list_ikf[ikf][i] for ikf in range(args.kfold)])})
        report_dict.update({"total_mask_active_weights_%_avg": statistics.mean([total_mask_active_weights_list_ikf[ikf][i] for ikf in range(args.kfold)]) / total_weights})
        report_dict.update({"total_active_weights_%_avg": statistics.mean([total_active_weights_list_ikf[ikf][i] for ikf in range(args.kfold)]) / total_weights})
        report_dict.update({"total_mask_fully_active_weights_%_avg": statistics.mean([total_mask_fully_active_weights_list_ikf[ikf][i] for ikf in range(args.kfold)]) / total_weights})
        if i % 5 == 0:
            report_dict.update({"train_accuracy_kf_avg": statistics.mean([accuracy_ikf[ikf][int(i / 5)] for ikf in range(args.kfold)]), 
                                "train_accuracy_b_kf_avg": statistics.mean([accuracy_b_ikf[ikf][int(i / 5)] for ikf in range(args.kfold)]), 
                                "train_f1_score_kf_avg": statistics.mean([f1_score_ikf[ikf][int(i / 5)] for ikf in range(args.kfold)]), 
                                "train_f1_score_b_kf_avg": statistics.mean([f1_score_b_ikf[ikf][int(i / 5)] for ikf in range(args.kfold)])})
            report_dict.update({"train_accuracy_kf_sd": statistics.stdev([accuracy_ikf[ikf][int(i / 5)] for ikf in range(args.kfold)]), 
                                "train_accuracy_b_kf_sd": statistics.stdev([accuracy_b_ikf[ikf][int(i / 5)] for ikf in range(args.kfold)]), 
                                "train_f1_score_kf_sd": statistics.stdev([f1_score_ikf[ikf][int(i / 5)] for ikf in range(args.kfold)]), 
                                "train_f1_score_b_kf_sd": statistics.stdev([f1_score_b_ikf[ikf][int(i / 5)] for ikf in range(args.kfold)])})
            if args.use_validation_set:
                report_dict.update({"val_accuracy_kf_avg": statistics.mean([accuracy_v_ikf[ikf][int(i / 5)] for ikf in range(args.kfold)]), 
                                    "val_accuracy_b_kf_avg": statistics.mean([accuracy_v_b_ikf[ikf][int(i / 5)] for ikf in range(args.kfold)]), 
                                    "val_f1_score_kf_avg": statistics.mean([f1_score_v_ikf[ikf][int(i / 5)] for ikf in range(args.kfold)]), 
                                    "val_f1_score_b_kf_avg": statistics.mean([f1_score_v_b_ikf[ikf][int(i / 5)] for ikf in range(args.kfold)])})
                report_dict.update({"val_accuracy_kf_sd": statistics.stdev([accuracy_v_ikf[ikf][int(i / 5)] for ikf in range(args.kfold)]), 
                                    "val_accuracy_b_kf_sd": statistics.stdev([accuracy_v_b_ikf[ikf][int(i / 5)] for ikf in range(args.kfold)]), 
                                    "val_f1_score_kf_sd": statistics.stdev([f1_score_v_ikf[ikf][int(i / 5)] for ikf in range(args.kfold)]), 
                                    "val_f1_score_b_kf_sd": statistics.stdev([f1_score_v_b_ikf[ikf][int(i / 5)] for ikf in range(args.kfold)])})
        if i == args.epoch - 1:
            report_dict.update({"test_accuracy_kf_avg": statistics.mean(test_accuracy_ikf), 
                                "test_accuracy_b_kf_avg": statistics.mean(test_accuracy_b_ikf), 
                                "test_f1_score_kf_avg": statistics.mean(test_f1_score_ikf), 
                                "test_f1_score_b_kf_avg": statistics.mean(test_f1_score_b_ikf)})
            report_dict.update({"test_accuracy_kf_sd": statistics.stdev(test_accuracy_ikf), 
                                "test_accuracy_b_kf_sd": statistics.stdev(test_accuracy_b_ikf), 
                                "test_f1_score_kf_sd": statistics.stdev(test_f1_score_ikf), 
                                "test_f1_score_b_kf_sd": statistics.stdev(test_f1_score_b_ikf)})
            report_dict.update({"run_time_kf_avg": statistics.mean(k_run_time)})
            report_dict.update({"run_time_kf_sd": statistics.stdev(k_run_time)})
            report_dict.update({"total_weights": total_weights})
            report_dict.update({"total_rules_sizes_kf_avg": statistics.mean(total_rules_sizes_ikf)})
            report_dict.update({"total_rules_sizes_kf_sd": statistics.stdev(total_rules_sizes_ikf)})

        wandb.log(report_dict)
        if args.hyperparameter_tuning:
            train.report(report_dict)

    wandb.log_artifact(args.folder_path)
    wandb.finish()


if __name__ == '__main__':
    default = {
        "kfold": 5,
        "use_validation_set": False,
        "epoch": 400,
        "batch_size": 128,
        "num_samples": 1,
        "random_binarization_rate": 0.75,
        "N": None,
        "use_not": False,
        "group_l0": True,
        "learning_rate": 5 * 10**-3 ,
        "lr_decay_rate": 0.75,
        "lr_decay_epoch": 100,
        "weight_decay": 10**-8,
        "lamba": 1.0,
        "droprate_init_input": 0.2,
        "droprate_init": 0.5,
        "beta_ema": 0.999,
        "temperature": 2./3.,
        "structure": '64',
        "use_bias": False
    }

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Arguments that will be passed or defaulted
    parser.add_argument('--project_name', type=str,
                        help='Name of the Wandb project')
    parser.add_argument('-ht', '--hyperparameter_tuning',action="store_true",
                        help='Whether a hyperparameter tuning will be or not performed')
    parser.add_argument('--num_cpu', type=int,
                        help='Num of cpus dedicated for hyperparameter tuning', default=multiprocessing.cpu_count())
    parser.add_argument('--num_gpu', type=int,
                        help='Num of gpus dedicated for hyperparameter tuning', default=torch.cuda.device_count())
    parser.add_argument('-d', '--data_set', type=str,
                        help='Set the data set for training. All the data sets in the dataset folder are available.')
    parser.add_argument('-k', '--kfold', type=int, default=argparse.SUPPRESS, help='Set the k of K-Folds cross-validation.')
    # parser.add_argument('-ki', '--ith_kfold', type=int, default=0, help='Do the i-th validation, 0 <= ki < k.')
    parser.add_argument('--use_validation_set', action="store_true",
                        help='Use the validation set for parameters tuning.')
    parser.add_argument('-e', '--epoch', type=int, default=argparse.SUPPRESS, help='Set the total epoch.')
    parser.add_argument('-bs', '--batch_size', type=int, default=argparse.SUPPRESS, help='Set the batch size.')
    parser.add_argument('-ns', '--num_samples', type=int, default=argparse.SUPPRESS, help='Number of samples for the hyperparameter tuning search')
    parser.add_argument('-p', '--random_binarization_rate',  type=float, default=argparse.SUPPRESS,
                        help='Random Binarization Rate for MLLP')
    parser.add_argument('-N', type=int, default=argparse.SUPPRESS,
                        help='L0 N parameter')
    parser.add_argument('--use_not', action="store_true",
                        help='Use the NOT (~) operator in logical rules.'
                             'It will enhance model capability but make the CRS more complex.')
    parser.add_argument('--group_l0', action="store_true",
                        help='L0 group_l0 parameter')
    parser.add_argument('--use_bias', action="store_true",
                        help='L0 use_bias parameter')

    # Arguments that will be passed or set up by the tuner
    parser.add_argument('-lr', '--learning_rate', type=float, default=argparse.SUPPRESS, help='Set the initial learning rate.')
    parser.add_argument('-lrdr', '--lr_decay_rate', type=float, default=argparse.SUPPRESS, help='Set the learning rate decay rate.')
    parser.add_argument('-lrde', '--lr_decay_epoch', type=int, default=argparse.SUPPRESS, help='Set the learning rate decay epoch.')
    parser.add_argument('-wd', '--weight_decay', type=float, default=argparse.SUPPRESS, help='Set the weight decay (L2 penalty).')
    parser.add_argument('--lamba', type=float, default=argparse.SUPPRESS,#1,
                        help='L0 Lamba parameter')
    parser.add_argument('--droprate_init_input', type=float, default=argparse.SUPPRESS,
                        help='L0 droprate_init_input parameter')
    parser.add_argument('--droprate_init', type=float, default=argparse.SUPPRESS,
                        help='L0 droprate_init parameter')
    parser.add_argument('--beta_ema', type=float, default=argparse.SUPPRESS,
                        help='L0 beta_ema parameter')
    parser.add_argument('--temperature', type=float, default=argparse.SUPPRESS,
                        help='L0 temperature parameter')
    parser.add_argument('-s', '--structure', type=str, default=argparse.SUPPRESS,
                        help='Set the structure of network. Only the number of nodes in middle layers are needed. '
                             'E.g., 64, 64_32_16. The total number of middle layers should be odd.')

    # Set logging
    # logging.basicConfig(level=logging.ERROR, stream=sys.stdout, format='[%(levelname)s] - %(message)s')

    # set seed
    torch.manual_seed(0)
    np.random.seed(0)
    
    args = parser.parse_args()

    print(args)

    config = {
        "project_name": args.project_name,
        "hyperparameter_tuning": args.hyperparameter_tuning,
        "data_set": args.data_set,
        "kfold": args.kfold if hasattr(args, "kfold") else default["kfold"],
        # "ith_kfold": args.ith_kfold,
        "use_validation_set": args.use_validation_set if args.use_validation_set else default["use_validation_set"],
        "epoch": args.epoch if hasattr(args, "epoch") else default["epoch"],
        "batch_size": args.batch_size if hasattr(args, "batch_size") else default["batch_size"],
        "num_samples": args.num_samples if hasattr(args, "num_samples") else default["num_samples"],
        "random_binarization_rate": args.random_binarization_rate if hasattr(args, "random_binarization_rate") else default["random_binarization_rate"],
        "N": args.N if hasattr(args, "N") else default["N"],
        "use_not": args.use_not if args.use_not else default["use_not"],
        "group_l0": args.group_l0 if args.group_l0 else default["group_l0"],
        "beta_ema": args.beta_ema if hasattr(args, "beta_ema") else default["beta_ema"], # if not args.hyperparameter_tuning else tune.quniform(0.05, 0.999, 0.001),
        "use_bias": args.use_bias if args.use_bias else default["use_bias"],
        "structure": args.structure if hasattr(args, "structure") else (default["structure"] if not args.hyperparameter_tuning else tune.choice(["32", "64", "128", "256", "32_32_32", "64_64_64", "128_128_128", "256_256_256"])),
        "learning_rate": args.learning_rate if hasattr(args, "learning_rate") else (default["learning_rate"] if not args.hyperparameter_tuning else tune.qloguniform(1e-4, 1e-1, 5e-5)),
        "lr_decay_rate": args.lr_decay_rate if hasattr(args, "lr_decay_rate") else (default["lr_decay_rate"] if not args.hyperparameter_tuning else tune.quniform(0.1, 1.0, 0.05)),
        "lr_decay_epoch": args.lr_decay_epoch if hasattr(args, "lr_decay_epoch") else (default["lr_decay_epoch"] if not args.hyperparameter_tuning else tune.randint(1, args.epoch - 1)),
        "weight_decay": args.weight_decay if hasattr(args, "weight_decay") else (default["weight_decay"] if not args.hyperparameter_tuning else tune.quniform(0.0, 0.1, 5e-5)),
        "lamba": args.lamba if hasattr(args, "lamba") else (default["lamba"] if not args.hyperparameter_tuning else tune.qloguniform(1e-4, 1.0, 5e-5)),
        "droprate_init_input": args.droprate_init_input if hasattr(args, "droprate_init_input") else (default["droprate_init_input"] if not args.hyperparameter_tuning else tune.quniform(0.01, 0.99, 0.01)),
        "droprate_init": args.droprate_init if hasattr(args, "droprate_init") else (default["droprate_init"] if not args.hyperparameter_tuning else tune.quniform(0.01, 0.99, 0.01)),
        "temperature": args.temperature if hasattr(args, "temperature") else (default["temperature"] if not args.hyperparameter_tuning else tune.quniform(0.01, 0.99, 0.01)),
    }

    data_path = os.path.join(os.path.join(os.path.dirname(__file__), DATA_DIR), args.data_set + '.data')
    info_path = os.path.join(os.path.join(os.path.dirname(__file__), DATA_DIR), args.data_set + '.info')

    if args.hyperparameter_tuning:
        trainable_with_cpu_gpu = tune.with_resources(partial(experiment, data_path=data_path, info_path=info_path), {"cpu": args.num_cpu, "gpu": args.num_gpu})
        tuner = tune.Tuner(trainable_with_cpu_gpu,
                        tune_config=tune.TuneConfig(
                                    num_samples=args.num_samples
                                    ),
                        run_config=RunConfig(failure_config=train.FailureConfig(max_failures=3)),
                        # run_config=RunConfig(
                        #             callbacks=[WandbLoggerCallback(project='l0_{}_k{}_ki{}_useValidationSet{}_e{}_bs{}_useNOT{}_N{}_local_rep{}_group_l0{}'.format(args.data_set, args.kfold, args.ith_kfold, args.use_validation_set, args.epoch, args.batch_size, args.use_not, args.N, args.local_rep, args.group_l0))]
                        #             ),
                        param_space=config)

        results = tuner.fit()

    else:
        experiment(args=config, data_path=data_path, info_path=info_path)

    # args.folder_name = 'l0_{}_k{}_ki{}_useValidationSet{}_e{}_bs{}_lr{}_lrdr{}_lrde{}_wd{}_useNOT{}_lamba{}_droprate_init_input{}_droprate_init{}_N{}_beta_ema{}_local_rep{}_temperature{}'.format(
    #     args.data_set, args.kfold, args.ith_kfold, args.use_validation_set, args.epoch, args.batch_size,
    #     args.learning_rate, args.lr_decay_rate, args.lr_decay_epoch, args.weight_decay, args.use_not, args.lamba, args.droprate_init_input, args.droprate_init, args.N, args.beta_ema, args.local_rep, args.temperature)

    # if not os.path.exists('log_folder'):
    #     os.mkdir('log_folder')
    # args.folder_name = args.folder_name + '_L' + args.structure
    # args.folder_path = os.path.join('log_folder', args.folder_name)
    # if not os.path.exists(args.folder_path):
    #     os.mkdir(args.folder_path)
    # args.model = os.path.join(args.folder_path, 'model.pth')
    # args.crs_file = os.path.join(args.folder_path, 'crs.txt')
    # args.plot_file = os.path.join(args.folder_path, 'plot_file.pdf')
    # args.log = os.path.join(args.folder_path, 'log.txt')
    # logging.basicConfig(level=logging.INFO, filename=args.log, filemode='w', format='[%(levelname)s] - %(message)s')
    # for arg in vars(args):
    #     print(arg, getattr(args, arg))
    # experiment(args)
