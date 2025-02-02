import os
num_numpy_threads = '16'
os.environ['OPENBLAS_NUM_THREADS'] = num_numpy_threads
os.environ['GOTO_NUM_THREADS'] = num_numpy_threads
os.environ['MKL_NUM_THREADS'] = num_numpy_threads
os.environ['NUMEXPR_NUM_THREADS'] = num_numpy_threads
os.environ['OMP_NUM_THREADS'] = num_numpy_threads


import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets import CIFAR10, MNIST, FashionMNIST, SVHN, CIFAR100, STL10, EMNIST, ImageFolder
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.model_selection import StratifiedKFold
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
from nc.utils import *

class StratifiedBatchSampler:
    """
    Stratified batch sampling
    Provides equal representation of target classes in each batch
    """
    def __init__(self, y, batch_size, shuffle=True):
        if torch.is_tensor(y):
            y = y.numpy()
        elif isinstance(y, list):
            y = np.array(y)
        assert len(y.shape) == 1, 'label array must be 1D'
        n_batches = int(len(y) / batch_size)
        self.skf = StratifiedKFold(n_splits=n_batches, shuffle=shuffle)
        self.X = torch.randn(len(y),1).numpy()
        self.y = y
        self.shuffle = shuffle
        self.n_batches = n_batches

    def __iter__(self):
        if self.shuffle:
            self.skf.random_state = torch.randint(0,int(1e8),size=()).item()
        for train_idx, test_idx in self.skf.split(self.X, self.y):
            yield test_idx

    def __len__(self):
        return self.n_batches


class IMBALANCECIFAR10(CIFAR10):
    cls_num = 10

    def __init__(self, root, imb_type='exp', imb_factor=0.01, rand_number=0, train=True,
                 transform=None, target_transform=None,
                 download=False):
        super(IMBALANCECIFAR10, self).__init__(root, train, transform, target_transform, download)
        np.random.seed(rand_number)
        img_num_list = self.get_img_num_per_cls(self.cls_num, imb_type, imb_factor)
        self.gen_imbalanced_data(img_num_list)

    def get_img_num_per_cls(self, cls_num, imb_type, imb_factor):
        img_max = len(self.data) / cls_num
        img_num_per_cls = []
        if imb_type == 'exp':
            for cls_idx in range(cls_num):
                num = img_max * (imb_factor**(cls_idx / (cls_num - 1.0)))
                img_num_per_cls.append(int(num))
        elif imb_type == 'step':
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max))
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max * imb_factor))
        else:
            img_num_per_cls.extend([int(img_max)] * cls_num)
        return img_num_per_cls

    def gen_imbalanced_data(self, img_num_per_cls):
        new_data = []
        new_targets = []
        targets_np = np.array(self.targets, dtype=np.int64)
        classes = np.unique(targets_np)
        # np.random.shuffle(classes)
        self.num_per_cls_dict = dict()
        for the_class, the_img_num in zip(classes, img_num_per_cls):
            self.num_per_cls_dict[the_class] = the_img_num
            idx = np.where(targets_np == the_class)[0]
            np.random.shuffle(idx)
            selec_idx = idx[:the_img_num]
            new_data.append(self.data[selec_idx, ...])
            new_targets.extend([the_class, ] * the_img_num)
        new_data = np.vstack(new_data)
        self.data = new_data
        self.targets = new_targets
        
    def get_cls_num_list(self):
        cls_num_list = []
        for i in range(self.cls_num):
            cls_num_list.append(self.num_per_cls_dict[i])
        return cls_num_list


class IMBALANCECIFAR100(CIFAR100):
    cls_num = 100

    def __init__(self, root, imb_type='exp', imb_factor=0.01, rand_number=0, train=True,
                 transform=None, target_transform=None,
                 download=False):
        super(IMBALANCECIFAR100, self).__init__(root, train, transform, target_transform, download)
        np.random.seed(rand_number)
        img_num_list = self.get_img_num_per_cls(self.cls_num, imb_type, imb_factor)
        self.gen_imbalanced_data(img_num_list)

    def get_img_num_per_cls(self, cls_num, imb_type, imb_factor):
        img_max = len(self.data) / cls_num
        img_num_per_cls = []
        if imb_type == 'exp':
            for cls_idx in range(cls_num):
                num = img_max * (imb_factor**(cls_idx / (cls_num - 1.0)))
                img_num_per_cls.append(int(num))
        elif imb_type == 'step':
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max))
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max * imb_factor))
        else:
            img_num_per_cls.extend([int(img_max)] * cls_num)
        return img_num_per_cls

    def gen_imbalanced_data(self, img_num_per_cls):
        new_data = []
        new_targets = []
        targets_np = np.array(self.targets, dtype=np.int64)
        classes = np.unique(targets_np)
        # np.random.shuffle(classes)
        self.num_per_cls_dict = dict()
        for the_class, the_img_num in zip(classes, img_num_per_cls):
            self.num_per_cls_dict[the_class] = the_img_num
            idx = np.where(targets_np == the_class)[0]
            np.random.shuffle(idx)
            selec_idx = idx[:the_img_num]
            new_data.append(self.data[selec_idx, ...])
            new_targets.extend([the_class, ] * the_img_num)
        new_data = np.vstack(new_data)
        self.data = new_data
        self.targets = new_targets
        
    def get_cls_num_list(self):
        cls_num_list = []
        for i in range(self.cls_num):
            cls_num_list.append(self.num_per_cls_dict[i])
        return cls_num_list




def make_synthetic_data(args):
    targets = torch.arange(0, args.num_classes).long()
    # create random labels from targets equal size

    targets = targets.repeat(args.dataset_size//args.num_classes)

    samples = torch.randn((args.dataset_size, args.feature_dim), dtype=torch.float32)
    
    # shuffle the one hot targets
    perm = torch.randperm(args.dataset_size)
    targets = targets[perm]

    return samples, targets

def make_synthetic_dataloader(args, features, targets):
    # create dataset
    train_dataset = TensorDataset(features, targets)

    # create dataloader
    if args.stratified_batch:
        train_loader = DataLoader(train_dataset, batch_sampler=StratifiedBatchSampler(targets, args.batch_size, shuffle=True), num_workers=2)
    else:
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    return train_loader

def get_mnist1d(path):
    return from_pickle(path)

def add_label_noise(dataset, noise_level):
    n = len(dataset)
    num_classes = len(torch.unique(dataset.targets))
    n_noisy = int(n * noise_level)
    noisy_indices = torch.randperm(n)[:n_noisy]
    noisy_labels = torch.randint(0, num_classes, (n_noisy,))
    dataset.targets[noisy_indices] = noisy_labels
    
    return dataset

def make_dataset(args):
    dataset_name, data_dir, batch_size, sample_size = args.dataset, args.data_dir, args.batch_size, args.sample_size
    if dataset_name == 'mnist1d':
        print('Dataset: MNIST1D.')
        dataset = get_mnist1d(data_dir + '/synthetic/MNIST1D/mnist1d_data.pkl')
        trainset = TensorDataset(torch.Tensor(dataset['x']), torch.LongTensor(dataset['y']))
        testset = TensorDataset(torch.Tensor(dataset['x_test']), torch.LongTensor(dataset['y_test']))
        trainset.targets = trainset.tensors[1]
        testset.targets = testset.tensors[1]
        if args.label_noise > 0:
            trainset = add_label_noise(trainset, args.label_noise)
        num_classes = 10
    elif dataset_name == 'mnist':
        print('Dataset: MNIST.')
        trainset = MNIST(root=data_dir, train=True, download=True, transform=transforms.Compose([
            transforms.Grayscale(3),
            transforms.Resize(32),
            transforms.RandomRotation(25),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.1307,), std=(0.3081,))
        ]))

        testset = MNIST(root=data_dir, train=False, download=True, transform=transforms.Compose([
            transforms.Grayscale(3),
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.1307,), std=(0.3081,))
        ]))
        num_classes = 10
    elif dataset_name == 'fashion-mnist':
        print('Dataset: FashionMNIST')
        trainset = FashionMNIST(root=data_dir, train=True, download=True, transform=transforms.Compose([
            transforms.Grayscale(3),
            transforms.Resize(32),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4, fill=0, padding_mode='constant'),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.2860,), std=(0.3530,))
        ]))

        testset = FashionMNIST(root=data_dir, train=False, download=True, transform=transforms.Compose([
            transforms.Grayscale(3),
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.2860,), std=(0.3530,))
        ]))
        num_classes = 10
    elif dataset_name == 'svhn':
        print('Dataset: SVHN')
        trainset = SVHN(root=data_dir, split='train', download=True, transform=transforms.Compose([
            transforms.RandomRotation(25),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4377, 0.4438, 0.4728], std=[0.1980, 0.2010, 0.1970])
        ]))

        testset = SVHN(root=data_dir, split='test', download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4377, 0.4438, 0.4728], std=[0.1980, 0.2010, 0.1970])
        ]))
        num_classes = 10
    elif dataset_name == 'cifar10':
        print('Dataset: CIFAR10.')
        
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4, fill=0, padding_mode='constant'),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
        ])
        
        if args.imb_factor < 1:
            print("We are running with CIFAR10LT")
            trainset = IMBALANCECIFAR10(root=data_dir, imb_factor=args.imb_factor, 
                                        train=True, download=True, transform=train_transform)
        else:
            trainset = CIFAR10(root=data_dir, train=True, download=True, transform=train_transform)

        testset = CIFAR10(root=data_dir, train=False, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
        ]))
        num_classes = 10
    elif dataset_name == 'emnist':
        trainset = EMNIST(root=data_dir, split='balanced', train=True, download=True, transform=transforms.Compose([
            transforms.Grayscale(3),
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.1751,), std=(0.3332,))
        ]))

        testset = EMNIST(root=data_dir, split='balanced', train=False, download=True, transform=transforms.Compose([
            transforms.Grayscale(3),
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.1751,), std=(0.3332,))
        ]))
        num_classes = 47
    elif dataset_name == 'cifar100':

        train_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, padding=4, fill=0, padding_mode='constant'),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5071, 0.4865, 0.4409], std=[0.2673, 0.2564, 0.2762]),
            ])
         
        if args.imb_factor < 1:
            print("We are running with CIFAR100LT")
            trainset = IMBALANCECIFAR100(root=data_dir, imb_factor=args.imb_factor, 
                                        train=True, download=True, transform=train_transform)
        else:
            trainset = CIFAR100(root=data_dir, train=True, download=True, transform=train_transform)

        testset = CIFAR100(root=data_dir, train=False, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5071, 0.4865, 0.4409], std=[0.2673, 0.2564, 0.2762])
        ]))
        num_classes = 100
    elif dataset_name == 'stl10':
        trainset = STL10(root=data_dir, split='train', download=True, transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(96, padding=4, fill=0, padding_mode='constant'),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4467, 0.4398, 0.4066], std=[0.2603, 0.2566, 0.2713]),
        ]))

        testset = STL10(root=data_dir, split='test', download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4467, 0.4398, 0.4066], std=[0.2603, 0.2566, 0.2713])
        ]))
        trainset.targets = trainset.labels
        testset.targets = testset.labels
        num_classes = 10
    elif dataset_name == 'tiny-imagenet':
        traindir = os.path.join(data_dir, 'TinyImageNet', 'train')
        valdir = os.path.join(data_dir, 'TinyImageNet', 'val')

        trainset = ImageFolder(traindir, transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(64, padding=8, fill=0, padding_mode='constant'),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4802, 0.4481, 0.3975], std=[0.2770, 0.2691, 0.2821]),
        ]))

        testset = ImageFolder(valdir, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4802, 0.4481, 0.3975], std=[0.2770, 0.2691, 0.2821])
        ]))
        num_classes = 200
    elif dataset_name == 'imagenet':
        traindir = os.path.join(data_dir, 'imagenet', 'train')
        valdir = os.path.join(data_dir, 'imagenet', 'val')

        trainset = ImageFolder(traindir, transform=transforms.Compose([
            transforms.Resize(256),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(224, padding=4, fill=0, padding_mode='constant'),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]))

        testset = ImageFolder(valdir, transform=transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]))
        num_classes = 1000
    else:
        raise ValueError

    if sample_size is not None:
        total_sample_size = num_classes * sample_size
        cnt_dict = dict()
        total_cnt = 0
        indices = []
        for i in range(len(trainset)):

            if total_cnt == total_sample_size:
                break

            label = trainset[i][1]
            if label not in cnt_dict:
                cnt_dict[label] = 1
                total_cnt += 1
                indices.append(i)
            else:
                if cnt_dict[label] == sample_size:
                    continue
                else:
                    cnt_dict[label] += 1
                    total_cnt += 1
                    indices.append(i)

        train_indices = torch.tensor(indices)
        trainloader = DataLoader(trainset, batch_size=batch_size, sampler=SubsetRandomSampler(train_indices), num_workers=1)
        if args.implicit_forward_bs is not None:
            implicit_trainloader = DataLoader(trainset, batch_size=args.implicit_forward_bs, sampler=SubsetRandomSampler(train_indices), num_workers=1)

    elif args.stratified_batch:
        trainloader = DataLoader(trainset, batch_sampler=StratifiedBatchSampler(trainset.targets, batch_size, shuffle=True), num_workers=int(num_numpy_threads))
        if args.implicit_forward_bs is not None:
            implicit_trainloader = DataLoader(trainset, batch_sampler=StratifiedBatchSampler(trainset.targets, args.implicit_forward_bs, shuffle=True), num_workers=int(num_numpy_threads))
 
    else:
        trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=int(num_numpy_threads))
        if args.implicit_forward_bs is not None:
            implicit_trainloader = DataLoader(trainset, batch_size=args.implicit_forward_bs, shuffle=True, num_workers=int(num_numpy_threads))

    if args.stratified_batch:
        testloader = DataLoader(testset, batch_sampler=StratifiedBatchSampler(testset.targets, batch_size, shuffle=False), num_workers=int(num_numpy_threads))
    else:
        testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=int(num_numpy_threads))
    
    if args.implicit_forward_bs is not None:
        return [trainloader, implicit_trainloader], testloader, num_classes
    return trainloader, testloader, num_classes
