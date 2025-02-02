
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lrs
from nc.optimisers.agd import AGD

# ------------------------------------------------------------------------------------------------------
# Loss Functions
# ------------------------------------------------------------------------------------------------------
def check_optimality_condition(H, W):
    # compute cosine similarity between H and W 
    # and check if it is close to 1
    # if not, then we are not at a local minimum
    # and we should continue training
    with torch.no_grad():
        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        H = H - torch.mean(H, dim=0, keepdim=True)
        W = W - torch.mean(W, dim=0, keepdim=True)
        cos_sim = cos(H, W)
        return torch.mean(cos_sim)

class DotRegressionLoss(nn.Module):
    def __init__(self) -> None:
        super(DotRegressionLoss, self).__init__()

    def forward(self, features=None, W=None, targets=None, b=None):
        Wc = W[targets, :] 
        bc = b[targets]
        # if fixed_ETF: remove the bias term from the loss
        # if fixed_ETF:
        #     bc = torch.zeros_like(bc)
            
        # change to einsum
        dot = torch.bmm(features.unsqueeze(1), Wc.unsqueeze(2)).view(-1) + bc.view(-1)
        loss = (1/2) * torch.mean(((dot - 1) ** 2))

        return loss
    
# ------------------------------------------------------------------------------------------------------
# Network Optimisation
# ------------------------------------------------------------------------------------------------------


def make_optimiser(args, my_model, momentum=0.9):

    if args.declarative_ETF:
        # create a trainable list of parameters (meaning only when requires_grad is True) excluding the classifier but not returning the parameter and not the name
        trainable = map(lambda x: x[1], filter(lambda p: p[1].requires_grad and p[0].find('classifier') < 0, my_model.named_parameters()))
    else:
        trainable = filter(lambda p: p.requires_grad, my_model.parameters())

    if args.sep_decay:
        wd_term = 0
    else:
        wd_term = args.weight_decay

    if args.optimiser == 'SGD':
        optimiser_function = optim.SGD
        kwargs = {'momentum': momentum,
                  'lr': args.lr,
                  'weight_decay': wd_term  # args.weight_decay
                  }
    elif args.optimiser == 'Adam':
        optimiser_function = optim.Adam
        kwargs = {
            'betas': (0.9, 0.999),
            'eps': 1e-08,
            'lr': args.lr,
            'weight_decay': wd_term  # args.weight_decay
        }
    elif args.optimiser == 'AdamW':
        optimiser_function = optim.AdamW
        kwargs = {
            'betas': (0.9, 0.999),
            'eps': 1e-08,
            'lr': args.lr,
            'weight_decay': wd_term  # args.weight_decay
        }
    elif args.optimiser == 'LBFGS':
        optimiser_function = optim.LBFGS
        kwargs = {'lr': args.lr,
                  'history_size': args.history_size,
                  'line_search_fn': 'strong_wolfe'
                  }
    elif args.optimiser == 'AGD':
        optimiser_function = AGD
        kwargs = {'gain': 10.0}
        trainable = list(trainable)
        
    return optimiser_function(trainable, **kwargs)


def make_scheduler(args, my_optimiser, dataloader=None):
    if args.decay_type == 'step':
        scheduler = lrs.StepLR(
            my_optimiser,
            step_size=args.patience,
            gamma=args.gamma
        )
    elif args.decay_type.find('step') >= 0:
        milestones = args.decay_type.split('_')
        milestones.pop(0)
        milestones = list(map(lambda x: int(x), milestones))
        scheduler = lrs.MultiStepLR(
            my_optimiser,
            milestones=milestones,
            gamma=args.gamma
        )
    elif args.decay_type == 'onecycle':    
        scheduler = lrs.OneCycleLR(
            my_optimiser,
            max_lr=args.max_lr,
            epochs=args.epochs - args.warmup_epochs,
            steps_per_epoch=len(dataloader),
            pct_start=0.3,
            anneal_strategy='cos',
            cycle_momentum=True,
            div_factor=args.div_factor,
            final_div_factor=args.final_div_factor,
            three_phase=True
        )
    return scheduler


def make_criterion(args):
    if args.loss == 'CrossEntropy':
        criterion = nn.CrossEntropyLoss()
    elif args.loss == 'MSE':
        criterion = nn.MSELoss()
    elif args.loss == 'DotRegression': 
        criterion = DotRegressionLoss()

    return criterion

# ------------------------------------------------------------------------------------------------------
# Metrics
# ------------------------------------------------------------------------------------------------------


class AverageMeter(object):
    """
    Computes and stores the average and current value
    Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def compute_accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    # probs = F.softmax(output, dim=1)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res, pred


def compute_NCC_match(features, num_classes, mu_c_dict, net_pred):

    batch_size = features.size(0)

    # Convert the dictionary to a tensor
    mu_c_tensor = torch.stack([mu_c_dict[c] for c in range(num_classes)])

    # Subtract the mean vectors from the features
    diff = features.unsqueeze(1) - mu_c_tensor.unsqueeze(0)

    # Compute the norm along the last dimension
    NCC_scores = torch.linalg.norm(diff, dim=-1)
    
    NCC_pred = torch.argmin(NCC_scores, dim=1)
    NCC_match_net = NCC_pred.eq(net_pred[0]).float().sum(0)
    return NCC_match_net.mul_(100.0 / batch_size)


