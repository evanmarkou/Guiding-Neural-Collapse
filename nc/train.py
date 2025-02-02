import os
num_numpy_threads = '16'
os.environ['OPENBLAS_NUM_THREADS'] = num_numpy_threads
os.environ['GOTO_NUM_THREADS'] = num_numpy_threads
os.environ['MKL_NUM_THREADS'] = num_numpy_threads
os.environ['NUMEXPR_NUM_THREADS'] = num_numpy_threads
os.environ['OMP_NUM_THREADS'] = num_numpy_threads


import sys
import pathlib
import numpy as np
import torch
import torch.nn as nn
import nc.models as models
from nc.utils import *
from nc.args import parse_train_args
from nc.datasets import make_dataset
from nc.models.model_structure import *
import time


def trainer(args, model, trainloader, epoch, criterion, optimiser, scheduler, logfile):

    losses = AverageMeter()
    cos_sims = AverageMeter()
    log_etas = AverageMeter()

    model.train()

    if scheduler is not None:
        print_and_save('\nTraining Epoch: [%d | %d] LR: %f' % (
            epoch + 1, args.epochs, scheduler.get_last_lr()[-1]), logfile)

    for batch_idx, (inputs, targets) in enumerate(trainloader):

        inputs, targets = inputs.to(args.device), targets.to(args.device)

      
        outputs = model(inputs, targets)
        
        if args.declarative_ETF:
            W = model.W
            b = model.b
        else:
            W = model.classifier.weight
            b = model.classifier.bias

        if args.loss == 'CrossEntropy':
            loss = criterion(outputs[0], targets)
            cos_sim = check_optimality_condition(outputs[1], W[targets, :])
        elif args.loss == 'DotRegression':
            loss = criterion(outputs[1], W, targets, b)
            cos_sim = check_optimality_condition(outputs[1], W[targets, :])
                    
        optimiser.zero_grad()
        loss.backward()

        if 'AGD' in args.optimiser:
            log_eta = optimiser.step()
        else:
            optimiser.step()

        if not args.declarative_ETF and not args.ETF_fc:
            model.normalise_classifier()    

        # record loss
        losses.update(loss.item(), inputs.size(0))
        cos_sims.update(cos_sim.item(), inputs.size(0))
        if 'AGD' in args.optimiser:
            log_etas.update(log_eta, inputs.size(0))

        [prec1, prec5], _ = compute_accuracy(outputs[0].detach().data, targets.detach().data, topk=(1, 5))

        if batch_idx % args.log_interval == 0:
            if 'AGD' in args.optimiser:
                print_and_save('[epoch: %d] (%d/%d)| Loss: %.4f | Cos. Sim.: %.4f | Top1 Batch Acc.: %.4f | Top5 Batch Acc.: %.4f | log(eta): %.4f' %
                        (epoch + 1, batch_idx + 1, len(trainloader), loss.item(), cos_sim.item(), prec1, prec5, log_eta), logfile)
            else:
                print_and_save('[epoch: %d] (%d/%d)| Loss: %.4f | Cos. Sim.: %.4f | Top1 Batch Acc.: %.4f | Top5 Batch Acc.: %.4f' %
                            (epoch + 1, batch_idx + 1, len(trainloader), loss.item(), cos_sim.item(), prec1, prec5), logfile)

        if batch_idx > 5 and args.debug:
            break
            
    if scheduler is not None:
        scheduler.step()

    return losses.avg, cos_sims.avg, log_etas.avg


def train(args, model, trainloader):

    criterion = make_criterion(args)
    optimiser = make_optimiser(args, model)
    if 'AGD' in args.optimiser:
        print('Initialising weights with orthogonal matrices')
        if args.optimiser == 'AGD':
            optimiser.init_weights()    
        scheduler = None
    else:
        # scheduler = make_scheduler(args, optimiser)
        scheduler = None

    log_path = args.save_path + '/logs/'
    epoch_checkpoints_path = args.save_path + '/epoch_checkpoints/'
    classifier_path = args.save_path + '/classifier/'
    metrics_path = args.save_path + '/metrics/'
    pathlib.Path(log_path).mkdir(exist_ok=True)
    pathlib.Path(epoch_checkpoints_path).mkdir(exist_ok=True)
    pathlib.Path(classifier_path).mkdir(exist_ok=True)
    pathlib.Path(metrics_path).mkdir(exist_ok=True)

    logfile = open(log_path + 'train_log.txt', 'a+') if args.resume else open(log_path + 'train_log.txt', 'w+')

    # load model if resuming training
    if args.resume:
        model.load_state_dict(torch.load(epoch_checkpoints_path + 'epoch_' + str(args.resume_checkpoint).zfill(3) + '.pt'))
        print_and_save('Resuming training from epoch ' + str(args.resume_checkpoint).zfill(3), logfile)
    else:
        print_and_save('Training from scratch', logfile)

    print_and_save('# of model parameters: ' +
                   str(count_network_parameters(args, model)), logfile)
    print_and_save(
        '--------------------- Training -------------------------------', logfile)


    loss = np.zeros(args.epochs)
    cos_sim = np.zeros(args.epochs)
    log_eta = np.zeros(args.epochs)

    for epoch in range(args.resume_checkpoint, args.epochs):

        metrics = trainer(args, model, trainloader, epoch,
                          criterion, optimiser, scheduler, logfile)

        loss[epoch], cos_sim[epoch], log_eta[epoch] = metrics

        print_separation_line(logfile)

        torch.save(model.state_dict(), epoch_checkpoints_path +
                   "epoch_" + str(epoch + 1).zfill(3) + ".pt")
        if args.declarative_ETF:
            torch.save(model.W, classifier_path +
                       "trained_weights_" + str(epoch + 1).zfill(3) + ".pt")
            torch.save(model.b, classifier_path +
                       "trained_bias_" + str(epoch + 1).zfill(3) + ".pt")
    
    np.save(metrics_path + "loss.npy", loss)
    np.save(metrics_path + "cos_sim.npy", cos_sim)
    np.save(metrics_path + "log_eta.npy", log_eta)
   
    logfile.close()


def main():

    args = parse_train_args()
    set_seed(manualSeed=args.seed)

    if args.optimiser == 'LBFGS':
        sys.exit('Support for training with 1st order methods!')

    device = torch.device("cuda:"+str(args.gpu_id)
                            if torch.cuda.is_available() else "cpu")
    args.device = device
    trainloader, _, args.num_classes = make_dataset(args)
    
    model = models.__dict__[args.model](num_classes=args.num_classes, args=args).to(device)

    train(args, model, trainloader)


if __name__ == "__main__":
    main()
