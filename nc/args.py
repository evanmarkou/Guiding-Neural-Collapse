import os
import datetime
import argparse

import torch
import numpy as np


def parse_train_args():
    parser = argparse.ArgumentParser()

    # Model Selection
    model_selection_group = parser.add_argument_group(
        description='Model Selection:')
    model_selection_group.add_argument('--model', type=str, default='resnet18')
    model_selection_group.add_argument('--no-bias', dest='bias',
                                       action='store_false')
    model_selection_group.add_argument('--ETF_fc', dest='ETF_fc',
                                       action='store_true')
    model_selection_group.add_argument('--fixdim', dest='fixdim',
                                       action='store_true')
    model_selection_group.add_argument('--declarative_ETF', dest='declarative_ETF', action='store_true')
    model_selection_group.add_argument('--log_interval', type=int, default=10, help='log interval for the UFM optimisation')

    # MLP settings
    mlp_settings_group = parser.add_argument_group(
        description='MLP Settings (only when using mlp and res_adapt(in which case only width has effect)):')
    mlp_settings_group.add_argument('--width', type=int, default=1024)
    mlp_settings_group.add_argument('--depth', type=int, default=6)

    # Hardware Settings
    hardware_setting_group = parser.add_argument_group(
        description='Hardware settings:')
    hardware_setting_group.add_argument('--gpu_id', type=int, default=0)
    hardware_setting_group.add_argument('--seed', type=int, default=23)
    hardware_setting_group.add_argument('--use_cudnn', action='store_true')

    # Directory Settings
    dir_setting_group = parser.add_argument_group(
        description='Directory settings:')
    dir_setting_group.add_argument('--dataset', type=str,
                                   choices=['mnist', 'fashion-mnist', 'svhn', 'cifar10', 'emnist',
                                            'cifar100', 'stl10', 'imagenet', 'synthetic', 'mnist1d'], default='mnist')
    dir_setting_group.add_argument('--imb_factor', type=float, default=1.0, help='Value indicating the imbalance ratio of the dataset.')
    dir_setting_group.add_argument('--storage', type=str, default='..')
    dir_setting_group.add_argument('--data_dir', type=str, default=None)
    dir_setting_group.add_argument('--uid', type=str, default=None)
    dir_setting_group.add_argument('--force', action='store_true',
                                   help='force to override the given uid')

    # Learning Options
    learning_options_group = parser.add_argument_group(
        description='Learning options:')
    learning_options_group.add_argument('--epochs', type=int, default=200,
                                        help='Max Epochs')
    learning_options_group.add_argument('--warmup_epochs', type=int, default=0,
                                        help='Number of epochs to have the weight classifier be learned explicitly, as a warmup stage')
    learning_options_group.add_argument('--batch_size', type=int,
                                        default=128, help='Batch size')
    learning_options_group.add_argument('--full_batch_mode', action='store_true',
                                        help='If True, use full batch mode')
    learning_options_group.add_argument('--implicit_forward_bs', type=int, 
                                        default=None, help='Batch size to run the second forward pass in the implicit case to accumulate features and statistics')
    learning_options_group.add_argument('--stratified_batch', action='store_true',
                                        help='If True, use stratified sampling for the batch')
    learning_options_group.add_argument('--loss', type=str, default='CrossEntropy',
                                        help='loss function configuration')
    learning_options_group.add_argument('--temperature', type=float, default=1,
                                        help='temperature for the feature normalisation')
    learning_options_group.add_argument('--sample_size', type=int,
                                        default=None, help='sample size PER CLASS')
    learning_options_group.add_argument('--num_classes', type=int, default=10,
                                        help='number of classes (related to the given dataset)')
    learning_options_group.add_argument('--label_noise', type=float, default=0, help='label noise level')
    learning_options_group.add_argument('--debug', action='store_true',
                                        help='If True, run few iterations for debugging purposes')
    learning_options_group.add_argument('--inference', type=bool, default=False, help='If True it will only run inference')
    learning_options_group.add_argument('--resume', action='store_true', help='If True it will resume training from the last checkpoint')
    learning_options_group.add_argument('--resume_checkpoint', type=int, default=0, help='Epoch to resume training from')

    # Optimisation specifications
    optim_specs_group = parser.add_argument_group(
        description='Optimisation specifications:')
    optim_specs_group.add_argument('--lr', type=float, default=0.05,
                                   help='learning rate')
    optim_specs_group.add_argument('--patience', type=int, default=40,
                                   help='learning rate decay per N epochs')
    optim_specs_group.add_argument('--decay_type', type=str,
                                   default='step', help='learning rate decay type')
    optim_specs_group.add_argument('--gamma', type=float, default=0.1,
                                   help='learning rate decay factor for step decay')
    optim_specs_group.add_argument('--div_factor', type=float, default=25, help='div factor for one cycle policy')
    optim_specs_group.add_argument('--final_div_factor', type=float, default=1e4, help='final div factor for one cycle policy')
    optim_specs_group.add_argument('--max_lr', type=float, default=0.25,
                                   help='max learning rate for one cycle policy')
    optim_specs_group.add_argument('--optimiser', default='SGD',
                                   help='optimiser to use')
    optim_specs_group.add_argument('--weight_decay', type=float,
                                   default=5e-4, help='weight decay')
    
    ufm_group = parser.add_argument_group(
        description='UFM specifications - adding weight decay on Features:')
    # The following two should be specified when testing adding wd on Features
    ufm_group.add_argument('--sep_decay', action='store_true',
                           help='whether to separate weight decay to last feature and last weights')
    ufm_group.add_argument('--feature_decay_rate', type=float,
                           default=1e-4, help='weight decay for last layer feature')

    # ETF distance specifications
    ETF_distance_group = parser.add_argument_group(
        description='ETF distance specifications:')
    ETF_distance_group.add_argument('--sanity', dest='sanity', action='store_true',
                                    help='If True, it will run a sanity check on the optimisation problem of measuring ETF distance')
    ETF_distance_group.add_argument('--skip', dest='skip', action='store_true',
                                    help='If True, it will skip the process of finding the closest ETFs')
    ETF_distance_group.add_argument('--reference', type=str, choices=['W', 'H'], 
                                    help='reference measurement to closest ETF: W=classifier weights, H=classifier features centred means')
    ETF_distance_group.add_argument('--ref_ETF', type=int, default=0,
                                    help='The reference ETF solution which will be compared against all other ETF solutions \
                        This value will reflect the ETF solution at the particular epoch. Number zero (0) indicates the ETF solution of the convex problem')
    ETF_distance_group.add_argument('--discrete_tol', type=float, default=-1,
                                    help='Level of tolerance between the ETF distance during the ETF discretisation process')
    ETF_distance_group.add_argument('--nc_metrics', type=str, choices=['train', 'test'], default='train', 
                                    help='Whether to measure NC of training or testing data')
    ETF_distance_group.add_argument('--tpt_threshold', type=float, default=99.9, help='Threshold for the start of the TPT phase')
    
    # UFM specifications
    ufm_group = parser.add_argument_group(description='UFM specifications:')
    ufm_group.add_argument('--dataset_size', type=int, default=50000, help='dataset size of the synthetic random features')
    ufm_group.add_argument('--feature_dim', type=int, default=512, help='feature dimension of the synthetic random features')

    args = parser.parse_args()

    if args.sanity:
        save_path = args.storage + '/experiments/' + str(args.uid)
    else:
        if args.uid is None:
            unique_id = str(np.random.randint(0, 100000))
            print("revise the unique id to a random number " + str(unique_id))
            args.uid = unique_id
            timestamp = datetime.datetime.now().strftime("%a-%b-%d-%H-%M")
            save_path = args.storage + '/experiments/' + args.dataset + '/' + args.model + \
                '/model_weights/' + args.uid + '-' + timestamp + '/' + args.nc_metrics
        else:
            save_path = args.storage + '/experiments/' + args.dataset + '/' + args.model + \
                '/model_weights/' + str(args.uid) + '/' + args.nc_metrics

    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    else:
        if not args.force:
            raise ("please use another uid ")
        else:
            print("override this uid" + args.uid)

    parser.add_argument('--save_path', default=save_path,
                        help='the output dir of weights')
    parser.add_argument('--arg', default=save_path +
                        '/args.txt', help='the args used')

    args = parser.parse_args()
    if args.data_dir is None:
        args.data_dir = args.storage + "/data"

    with open(args.arg, 'w') as f:
        print(args)
        print(args, file=f)
        f.close()
    if args.use_cudnn:
        print("cudnn is used")
        torch.backends.cudnn.benchmark = True
    else:
        print("cudnn is not used")
        torch.backends.cudnn.benchmark = False

    return args


def parse_eval_args():
    parser = argparse.ArgumentParser()

    # parameters
    # Model Selection
    model_selection_group = parser.add_argument_group(
        description='Model Selection:')
    model_selection_group.add_argument('--model', type=str, default='resnet18')
    model_selection_group.add_argument('--no-bias', dest='bias',
                                       action='store_false')
    model_selection_group.add_argument('--ETF_fc', dest='ETF_fc',
                                       action='store_true')
    model_selection_group.add_argument('--fixdim', dest='fixdim',
                                        action='store_true')
    model_selection_group.add_argument('--declarative_ETF', dest='declarative_ETF', action='store_true')
    model_selection_group.add_argument('--log_interval', type=int, default=10, help='log interval for the UFM optimisation')

    # MLP settings (only when using mlp and res_adapt(in which case only width has effect))
    mlp_settings_group = parser.add_argument_group(
        description='MLP Settings (only when using mlp and res_adapt(in which case only width has effect)):')
    mlp_settings_group.add_argument('--width', type=int, default=1024)
    mlp_settings_group.add_argument('--depth', type=int, default=6)

    # Hardware Settings
    hardware_setting_group = parser.add_argument_group(
        description='Hardware settings:')
    hardware_setting_group.add_argument('--gpu_id', type=int, default=0)
    hardware_setting_group.add_argument('--seed', type=int, default=23)
    hardware_setting_group.add_argument('--use_cudnn', action='store_true')

    # Directory Settings
    dir_setting_group = parser.add_argument_group(
        description='Directory settings:')
    dir_setting_group.add_argument('--dataset', type=str,
                                   choices=['mnist', 'fashion-mnist', 'svhn', 'cifar10', 'emnist',
                                            'cifar100', 'stl10', 'imagenet', 'synthetic', 'mnist1d'], default='mnist')
    dir_setting_group.add_argument('--imb_factor', type=float, default=1.0, help='Value indicating the imbalance ratio of the dataset.')
    dir_setting_group.add_argument('--storage', type=str, default='..')
    dir_setting_group.add_argument('--data_dir', type=str, default=None)
    dir_setting_group.add_argument('--load_path', type=str, default=None)
    dir_setting_group.add_argument('--uid', type=str, default=None, help='the uid of the model to be loaded')
    dir_setting_group.add_argument('--uid2', type=str, default=None, help='the uid of the second model to be loaded for comparison. Usually reserved for standard models')
    dir_setting_group.add_argument('--uid3', type=str, default=None, help='the uid of the third model to be loaded for comparison. Usually reserved for fixed ETF models')

    dir_setting_group.add_argument('--force', action='store_true',
                                   help='force to override the given uid')

    # Learning Options
    learning_options_group = parser.add_argument_group(
        description='Learning options:')
    learning_options_group.add_argument('--epochs', type=int, default=200,
                                        help='Max Epochs')
    learning_options_group.add_argument('--batch_size', type=int,
                                        default=128, help='Batch size')
    learning_options_group.add_argument('--temperature', type=float, default=1,
                                        help='temperature for the feature normalisation')
    learning_options_group.add_argument('--full_batch_mode', action='store_true',
                                        help='If True, use full batch mode')
    learning_options_group.add_argument('--implicit_forward_bs', type=int, 
                                        default=None, help='Batch size to run the second forward pass in the implicit case to accumulate features and statistics')
    learning_options_group.add_argument('--stratified_batch', action='store_true',
                                        help='If True, use stratified sampling for the batch')
    learning_options_group.add_argument('--sample_size', type=int,
                                        default=None, help='sample size PER CLASS')
    learning_options_group.add_argument('--num_classes', type=int, default=10,
                                        help='number of classes (related to the given dataset)')
    learning_options_group.add_argument('--label_noise', type=float, default=0, help='label noise level')
    learning_options_group.add_argument('--debug', action='store_true',
                                        help='If True, run few iterations for debugging purposes')
    learning_options_group.add_argument('--inference', type=bool, default=True, help='If True it will only run inference')

    # Neural Collapse Measurement Settings
    nc_measure_group = parser.add_argument_group(
        description='Neural Collapse Measurement Settings:')
    nc_measure_group.add_argument('--nc_metrics', type=str,
                                  choices=['train', 'test'], default='train', help='Whether to measure NC of training or testing data')
    nc_measure_group.add_argument('--skip', dest='skip', action='store_true',
                                  help='If True, it will skip the process of finding the closest ETFs')
    nc_measure_group.add_argument('--reference', type=str, nargs='+',
                                  help='reference measurement to closest ETF: W=classifier weights, H=classifier features centred means')
    nc_measure_group.add_argument('--ref_ETF', type=int, default=0,
                                  help='The reference ETF solution which will be compared against all other ETF solutions \
                        This value will reflect the ETF solution at the particular epoch. Number zero (0) indicates the ETF solution of the convex problem')
    nc_measure_group.add_argument('--tpt_threshold', type=float, default=99.9, help='Threshold for the start of the TPT phase')
   
     # UFM specifications
    ufm_group = parser.add_argument_group(description='UFM specifications:')
    ufm_group.add_argument('--dataset_size', type=int, default=50000, help='dataset size of the synthetic random features')
    ufm_group.add_argument('--feature_dim', type=int, default=512, help='feature dimension of the synthetic random features')
    
    args = parser.parse_args()

    if args.data_dir == None:
        args.data_dir = args.storage + '/data'

    if args.load_path == None:
        args.load_path = args.storage + '/experiments/' + args.dataset + '/' + args.model + \
            '/model_weights/' + str(args.uid) + "/" + args.nc_metrics

    return args

