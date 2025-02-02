import os
import random
import torch
import numpy as np
import json
import pickle
import logging
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


class Params:
    """
    Class that loads hyperparameters from a json file
    """

    def __init__(self, json_path):
        self.update(json_path)

    def save(self, json_path):
        """
        Saves parameters to json file
        :param json_path: the actual json path
        """
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self, json_path):
        """
        Loads parameters from json file
        :param json_path: the actual json path
        """
        with open(json_path, 'r') as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        """
        Gives dict-like access to Params instance by `params.dict['learning_rate']`
        :return: Dictionary
        """
        return self.__dict__


def set_logger(log_path, name=None, mpi=False):
    """
    Sets the logger to log info in terminal and file `log_path`
    :param log_path: The log path directory
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s : %(levelname)s : %(message)s'))
        logger.addHandler(file_handler)

        # Logging to the console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)

    return logger


def save_dict_to_json(d, json_path):
    """
    Saves dictionary of floats to json file
    :param d: dictionary of float-castable values
    :param json_path: path to json file
    """
    with open(json_path, 'w') as f:
        # Require conversion of the values to float. It doesn't accept np.array or np.float
        d = {k: float(v) for k, v in d.items()}
        json.dump(d, f, indent=4)


def set_seed(manualSeed=666):
    random.seed(manualSeed)
    np.random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    torch.cuda.manual_seed(manualSeed)
    torch.cuda.manual_seed_all(manualSeed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(manualSeed)


def to_pickle(thing, path): # save something
    with open(path, 'wb') as handle:
        pickle.dump(thing, handle, protocol=3)


def from_pickle(path): # load something
    thing = None
    with open(path, 'rb') as handle:
        thing = pickle.load(handle)
    return thing

def print_and_save(text_str, file_stream, save_only=False):
    if not save_only:
        print(text_str)
    print(text_str, file=file_stream)


def print_separation_line(logfile, times=1, symbol='-', save_only=False):
    for _ in range(times):
        print_and_save(f'{symbol}'*50, logfile, save_only)


def count_network_parameters(args, model):
    if args.declarative_ETF:
        parameters = map(lambda x: x[1], filter(lambda p: p[1].requires_grad and p[0].find('classifier') < 0, model.named_parameters()))
    else:
        parameters = filter(lambda p: p.requires_grad, model.parameters())
    return sum([np.prod(p.size()) for p in parameters])

def plot_grad_flow(args, epoch, model, features=None):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.
    
    Usage: Plug this function in Trainer class after loss.backwards() as 
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    named_parameters = model.named_parameters()
    named_parameters = list(named_parameters)
    if features is not None:
            named_parameters.append(('features', features))
    if args.declarative_ETF:       
        named_parameters.append(('feature_mean', model.ClosestETFGeometry.feature_mean))
        named_parameters.append(('W', model.W))
        named_parameters.append(('b', model.b))

        
    ave_grads = []
    max_grads= []
    layers = []
    f = open(f'grad-flow/epoch_{epoch}_grad_flow.txt', 'w')
    for n, p in named_parameters:
        if(p.requires_grad):
            layers.append(n)
            f.write(f"Layer: {n} | Size: {p.size()} | Grad: {p.grad}\n | GradMean: {p.grad.abs().mean().item()}\n | GradMax: {p.grad.abs().max().item()}\n\n")
            ave_grads.append(p.grad.abs().mean().item())
            max_grads.append(p.grad.abs().max().item())
    f.close()
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom = -0.001, top=0.02) # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
    plt.savefig(f'grad-flow/epoch_{epoch}_grad_flow.png', bbox_inches='tight', dpi=120)