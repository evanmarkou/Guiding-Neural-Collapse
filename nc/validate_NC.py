import os
num_numpy_threads = '16'
os.environ['OPENBLAS_NUM_THREADS'] = num_numpy_threads
os.environ['GOTO_NUM_THREADS'] = num_numpy_threads
os.environ['MKL_NUM_THREADS'] = num_numpy_threads
os.environ['NUMEXPR_NUM_THREADS'] = num_numpy_threads
os.environ['OMP_NUM_THREADS'] = num_numpy_threads

import sys
import pickle
import pathlib
from tqdm import tqdm

import numpy as np
import torch
import scipy.linalg as scilin

import nc.models as models
from nc.utils import *
from nc.models.model_structure import *
from nc.args import parse_eval_args
from nc.datasets import make_dataset

class FCFeatures:
    def __init__(self):
        self.outputs = []

    def __call__(self, module, module_in):
        self.outputs.append(module_in)

    def clear(self):
        self.outputs = []


def split_array(input_array, batchsize=128):
    input_size = input_array.shape[0]
    num_splits, res_splits = input_size // batchsize, input_size % batchsize
    output_array_list = list()
    if res_splits == 0:
        output_array_list = np.split(input_array, batchsize, axis=0)
    else:
        for i in range(num_splits):
            output_array_list.append(
                input_array[i * batchsize:(i + 1) * batchsize])

        output_array_list.append(input_array[num_splits * batchsize:])

    return output_array_list


def compute_info(args, model, fc_features, dataloader):
    num_data = 0
    mu_G = 0
    mu_c_dict = dict()
    num_class_dict = dict()
    before_class_dict = dict()
    top1 = AverageMeter()
    top5 = AverageMeter()
    NCC_match = AverageMeter()

    for computation in ['Mean', 'Pred']:
        for batch_idx, (inputs, targets) in tqdm(enumerate(dataloader), total=len(dataloader), 
                                                 desc=f'Computing {computation} dataset loop...'):

            inputs, targets = inputs.to(args.device), targets.to(args.device)

            with torch.no_grad():
                outputs, features = model(inputs, targets)

            if computation == 'Mean':
                mu_G += torch.sum(features, dim=0)

                for b in range(len(targets)):
                    y = targets[b].item()
                    if y not in mu_c_dict:
                        mu_c_dict[y] = features[b, :]
                        before_class_dict[y] = [
                            features[b, :].detach().cpu().numpy()]
                        num_class_dict[y] = 1
                    else:
                        mu_c_dict[y] += features[b, :]
                        before_class_dict[y].append(
                            features[b, :].detach().cpu().numpy())
                        num_class_dict[y] = num_class_dict[y] + 1

            num_data += targets.shape[0]

            if computation == 'Pred':
                [prec1, prec5], net_pred = compute_accuracy(
                    outputs.detach().data, targets.detach().data, topk=(1, 5))
                top1.update(prec1.item(), inputs.size(0))
                top5.update(prec5.item(), inputs.size(0))

                NCC_net = compute_NCC_match(features.detach().data, args.num_classes, mu_c_dict, net_pred)

                NCC_match.update(NCC_net.item(), inputs.size(0))

            if args.debug and batch_idx > 20:
                break

        if computation == 'Mean':
            mu_G /= num_data
            for i in range(len(mu_c_dict.keys())):
                mu_c_dict[i] /= num_class_dict[i]

    return mu_G, mu_c_dict, before_class_dict, None, top1.avg, top5.avg, 100 - NCC_match.avg


def compute_Sigma_W(args, before_class_dict, mu_c_dict, batchsize=512):
    num_data = 0
    Sigma_W = 0
    for target in before_class_dict.keys():
        class_feature_list = split_array(np.array(before_class_dict[target]), batchsize=batchsize)
        for i in range(len(class_feature_list)):
            features = torch.from_numpy(class_feature_list[i]).to(args.device)
            h = features - mu_c_dict[target].unsqueeze(0)
            Sigma_W += torch.einsum('bi, bj->ij', h, h)
            num_data += features.shape[0]
   
    Sigma_W /= num_data
    return Sigma_W.detach().cpu().numpy()

def compute_Sigma_B(mu_c_dict, mu_G):
    Sigma_B = 0
    K = len(mu_c_dict)
    for i in range(K):
        Sigma_B += (mu_c_dict[i] -
                    mu_G).unsqueeze(1) @ (mu_c_dict[i] - mu_G).unsqueeze(0)

    Sigma_B /= K

    return Sigma_B.cpu().numpy()

def compute_Variability_Collapse_index(Sigma_W, Sigma_B, K):
    Sigma_T = Sigma_B + Sigma_W
    return 1 - np.trace(scilin.pinv(Sigma_T) @ Sigma_B) / min(Sigma_B.shape[0], K - 1)


def compute_ETF(W, device):
    K = W.shape[0]
    WWT = torch.mm(W, W.T)
    WWT /= torch.linalg.norm(WWT, ord='fro')

    sub = (torch.eye(K) - 1 / K * torch.ones((K, K))).to(device) / pow(K - 1, 0.5)
    ETF_metric = torch.linalg.norm(WWT - sub, ord='fro')
    return ETF_metric.detach().cpu().numpy().item()


def compute_W_H_relation(W, mu_c_dict, mu_G, device):
    K = len(mu_c_dict)
    H = torch.empty(mu_c_dict[0].shape[0], K)
    for i in range(K):
        H[:, i] = mu_c_dict[i] - mu_G

    WH = torch.mm(W, H.to(device))
    WH /= torch.linalg.norm(WH, ord='fro')
    sub = 1 / pow(K - 1, 0.5) * (torch.eye(K) - 1 / K * torch.ones((K, K))).to(device)

    res = torch.linalg.norm(WH - sub, ord='fro')
    return res.detach().cpu().numpy().item(), H


def compute_Wh_b_relation(W, mu_G, b, device):
    Wh = torch.mv(W, mu_G.to(device))
    res_b = torch.linalg.norm(Wh + b.to(device), ord=2)
    return res_b.detach().cpu().numpy().item()


def compute_W_H_equinorm(W, H):
    H_norms = torch.linalg.norm(H, dim=0)
    W_norms = torch.linalg.norm(W.T, dim=0)

    H_equinorm = (torch.std(H_norms) / torch.mean(H_norms)
                  ).detach().cpu().numpy().item()
    W_equinorm = (torch.std(W_norms) / torch.mean(W_norms)
                  ).detach().cpu().numpy().item()

    abs_diff = np.abs(H_equinorm - W_equinorm)
    return W_equinorm, H_equinorm, abs_diff


def compute_numerical_rank(all_features):
    nucl_fro_metric_list = [] # r = nuclear/frobenius
    for i in all_features:
        class_feature = np.array(all_features[i])
        s = scilin.svd(class_feature, compute_uv=False)  # s is all singular values
        nuclear_norm = np.sum(s)
        frobenius_norm = np.linalg.norm(class_feature, ord='fro')
        nucl_fro_metric_class = (nuclear_norm / frobenius_norm) ** 2
        nucl_fro_metric_list.append(nucl_fro_metric_class)
    nucl_fro_metric = np.mean(nucl_fro_metric_list)
    return nucl_fro_metric


def compute_margin(args, before_class_dict, W, mu_G, batchsize=128):
    num_data = 0
    avg_cos_margin = 0
    all_cos_margin = list()

    W = W - torch.mean(W, dim=0, keepdim=True)

    for target in before_class_dict.keys():
        class_features_list = split_array(np.array(before_class_dict[target]), batchsize=batchsize)
        for i in range(len(class_features_list)):
            features = torch.from_numpy(class_features_list[i]).to(args.device)

            centred_features = features - mu_G.unsqueeze(0)
            cos_outputs = (centred_features @ W.T) / (
                torch.linalg.norm(centred_features, dim=1, keepdim=True) * torch.linalg.norm(W.T, dim=0, keepdim=True))

            false_cos_outputs = cos_outputs.clone()
            false_cos_outputs[:, target] = -np.inf
            false_cos_targets = torch.argmax(false_cos_outputs, dim=1)

            cos_margin = cos_outputs[:, target] - torch.gather(false_cos_outputs, 1, false_cos_targets.unsqueeze(1)).reshape(-1)
            all_cos_margin.append(cos_margin.detach().cpu().numpy())
            avg_cos_margin += torch.sum(cos_margin)

            num_data += features.shape[0]

    avg_cos_margin /= num_data
    all_cos_margin = np.sort(np.concatenate(all_cos_margin, axis=0))

    return avg_cos_margin.item(), all_cos_margin


def main():
    args = parse_eval_args()
    set_seed(manualSeed=args.seed)

    if args.load_path is None:
        sys.exit('Need to input the path to a pre-trained model!')

    device = torch.device("cuda:" + str(args.gpu_id)
                          if torch.cuda.is_available() else "cpu")
    args.device = device

    trainloader, testloader, args.num_classes = make_dataset(args)


    if args.nc_metrics == 'train':
        dataloader = trainloader            
    else:
        dataloader = testloader

    model = models.__dict__[args.model](num_classes=args.num_classes, args=args).to(device)

    metric_info_dict = {
        'collapse_metric': [],
        'ETF_metric': [],
        'WH_relation_metric': [],
        'Wh_b_relation_metric': [],
        'WH_equinorm_diff_metric': [],
        'W_equinorm_metric': [],
        'H_equinorm_metric': [],
        'numerical_rank_metric': [],
        'avg_cos_margin': [],
        'all_cos_margin': [],
        'acc1': [],
        'acc5': [],
        'NCC_mismatch': []
    }

    data_info_dict = {
        'Sigma_B': [],
        'Sigma_W': [],
        'mu_G': [],
        'mu_c_dict': []
    }

    variable_info_dict = {
        'W' : [],
        'b' : [],
        'H' : []
    }

    logs_path = args.load_path + '/logs'
    pathlib.Path(logs_path).mkdir(parents=True, exist_ok=True)
    file = f'{logs_path}/{args.nc_metrics}_NC_metrics.txt' \
        if not args.debug else f'{logs_path}/{args.nc_metrics}_NC_metrics-DEBUG.txt'
    logfile = open(file, 'w')

    for epoch in range(0, args.epochs, args.log_interval):
        print_separation_line(logfile, 2, symbol='=')
        print_and_save(
            f"Measure {args.nc_metrics} NC metrics for epoch {epoch + 1}", logfile)
        print_separation_line(logfile, 2, symbol='=')

        load_path = args.load_path.replace('test', 'train') if args.nc_metrics == 'test' else args.load_path

        model.load_state_dict(torch.load(load_path + '/epoch_checkpoints/epoch_' + str(epoch + 1).zfill(3) + '.pt'))
        if args.declarative_ETF:
            model.W = torch.load(load_path + '/classifier/trained_weights_' + str(epoch + 1).zfill(3) + '.pt')
            model.b = torch.load(load_path + '/classifier/trained_bias_' + str(epoch + 1).zfill(3) + '.pt')
            
        model.eval()

        for n, p in model.named_parameters():
            if 'classifier.weight' in n:
                W = p
            if 'classifier.bias' in n:
                b = p
        if args.declarative_ETF:
            W = model.W
            b = model.b

        print_and_save(
            f"Computing per-class means (mu_c), global mean (mu_G), top-1, top-5 accuracies, and NCC mismatch...", logfile)
        mu_G, mu_c_dict, before_class_dict, _, acc1, acc5, NCC_mismatch = compute_info(args, model, None, dataloader)
        print_separation_line(logfile)

       

        print_and_save(
            f"Computing the within-class covariance, Sigma_W...", logfile)
        Sigma_W = compute_Sigma_W(args, before_class_dict, mu_c_dict, batchsize=args.batch_size)
        print_and_save(
            f"Within-class covariance Sigma_W:\n{Sigma_W}", logfile)
        print_separation_line(logfile)

        print_and_save(
            f"Computing the between-class covariance, Sigma_B...", logfile)
        Sigma_B = compute_Sigma_B(mu_c_dict, mu_G)
        print_and_save(
            f"Between-class covariance Sigma_B:\n{Sigma_B}", logfile)

        print_separation_line(logfile)

        print_and_save("Computing the covariance collapse: NC1...", logfile)

        try:
            Sigma_B_inv = scilin.pinv(Sigma_B)
        except np.linalg.LinAlgError as e:
            if "SVD did not converge" in str(e):
                # Adding regularization to Sigma_B
                regularized_Sigma_B = Sigma_B + np.eye(Sigma_B.shape[0]) * 1e-4
                Sigma_B_inv = scilin.pinv(regularized_Sigma_B)
            else:
                raise  # Re-raise the exception if it's not related to SVD convergence

        collapse_metric = np.trace(Sigma_W @ Sigma_B_inv) / len(mu_c_dict)
        print_and_save(f"NC1: {collapse_metric}", logfile)

        print_separation_line(logfile)

        print_and_save("Computing the variability collapse Index: VCI...", logfile)
        VCI = compute_Variability_Collapse_index(Sigma_W, Sigma_B, len(mu_c_dict))
        print_and_save(f"VCI: {VCI}", logfile)

        print_separation_line(logfile)

        print_and_save("Computing numerical rank: NC1...", logfile)
        nucl_fro_metric = compute_numerical_rank(before_class_dict)
        print_and_save(f"NC1(Numerical rank = Nuclear/Frobenius): {nucl_fro_metric}", logfile)

        print_separation_line(logfile)

        print_and_save("Computing ETF convergence: NC2...", logfile)
        ETF_metric = compute_ETF(W, device)
        print_and_save(f"NC2: {ETF_metric}", logfile)

        print_separation_line(logfile)

        print_and_save("Computing self-duality convergence: NC3...", logfile)
        WH_relation_metric, H = compute_W_H_relation(W, mu_c_dict, mu_G, device)
        print_and_save(f"NC3: {WH_relation_metric}", logfile)

        print_separation_line(logfile)

        print_and_save("Computing the collapse of bias: NC4...", logfile)
        if args.bias or args.declarative_ETF or args.ETF_fc:
            Wh_b_relation_metric = compute_Wh_b_relation(W, mu_G, b, device)
        else:
            Wh_b_relation_metric = compute_Wh_b_relation(
                W, mu_G, torch.zeros((W.shape[0], )), device)
        print_and_save(f"NC4: {Wh_b_relation_metric}", logfile)

        print_separation_line(logfile)

        print_and_save("Computing equinorm: NC2...", logfile)
        W_equinorm, H_equinorm, WH_equinorm_diff = compute_W_H_equinorm(W, H)
        print_and_save(f"NC2(W-equinorm): {W_equinorm}", logfile)
        print_and_save(f"NC2(H-equinorm): {H_equinorm}", logfile)
        print_and_save(f"NC2(WH-equinorm-diff): {WH_equinorm_diff}", logfile)

        print_separation_line(logfile)

        print_and_save("Computing cosine margins: NC2...", logfile)
        avg_cos_margin, all_cos_margin = compute_margin(args, before_class_dict, W, mu_G,
                                                        batchsize=args.batch_size)
        print_and_save(f"NC2(Average cosine margin): {avg_cos_margin}", logfile)

        data_info_dict['Sigma_W'].append(Sigma_W)
        data_info_dict['Sigma_B'].append(Sigma_B)
        metric_info_dict['collapse_metric'].append(collapse_metric)
        metric_info_dict['ETF_metric'].append(ETF_metric)
        metric_info_dict['WH_relation_metric'].append(WH_relation_metric)
        metric_info_dict['Wh_b_relation_metric'].append(Wh_b_relation_metric)
        metric_info_dict['W_equinorm_metric'].append(W_equinorm)
        metric_info_dict['H_equinorm_metric'].append(H_equinorm)
        metric_info_dict['WH_equinorm_diff_metric'].append(WH_equinorm_diff)
        metric_info_dict['numerical_rank_metric'].append(nucl_fro_metric)
        metric_info_dict['avg_cos_margin'].append(avg_cos_margin)
        metric_info_dict['all_cos_margin'].append(all_cos_margin)

        variable_info_dict['W'].append((W.detach().cpu().numpy()))
        if args.bias:
            variable_info_dict['b'].append(b.detach().cpu().numpy())
        variable_info_dict['H'].append(H.detach().cpu().numpy())

        data_info_dict['mu_G'].append(mu_G.detach().cpu().numpy())
        data_info_dict['mu_c_dict'].append(mu_c_dict)

        metric_info_dict['acc1'].append(acc1)
        metric_info_dict['acc5'].append(acc5)
        metric_info_dict['NCC_mismatch'].append(NCC_mismatch)

        print_separation_line(logfile)
        print_and_save('[epoch: %d] | %s top-1 accuracy: %.4f | %s top-5 accuracy: %.4f | %s NCC-mismatch: %.4f' %
                        (epoch + 1, args.nc_metrics, acc1, args.nc_metrics,
                        acc5, args.nc_metrics, NCC_mismatch), logfile)

    logfile.close()

    path = args.load_path + '/metrics/'
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)

    metric_file = path + \
        f'{args.nc_metrics}_NC_metrics.pkl' if not args.debug else args.load_path + \
        f'{args.nc_metrics}_NC_metrics-DEBUG.pkl'
    with open(metric_file, 'wb') as f:
        pickle.dump(metric_info_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

    variable_file = path + f'{args.nc_metrics}_NC_variables.pkl'
    with open(variable_file, 'wb') as f:
        pickle.dump(variable_info_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    # Don't really need this!
    # data_file = path + f'{args.nc_metrics}_NC_data.pkl'
    # with open(data_file, 'wb') as f:
    #     pickle.dump(data_info_dict, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()
