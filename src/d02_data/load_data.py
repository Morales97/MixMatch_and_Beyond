import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np


def get_dataloaders(path='../../data', batch_size=64):
    transform_train = transforms.Compose(
        # transforms.RandomCrop(32, padding=4),
        # transforms.RandomHorizontalFlip(),
        [transforms.ToTensor(),
         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    batch_size = batch_size

    trainset = datasets.CIFAR10(root=path, train=True,
                                download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=2)

    testset = datasets.CIFAR10(root=path, train=False,
                               download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    return trainset, trainloader, testset, testloader


def get_dataloaders_validation(path="../../data", batch_size=64, shuffle=False, augment=False,
                               train_size=45000, val_size=5000):
    """
    Include split in train and valdiation set
    """
    assert ((train_size >= 0) and (val_size >= 0) and (train_size + val_size <= 50000))

    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010],
    )

    if augment:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    # Need to load dataset twice in case train data is augmented but not validation
    train_set = datasets.CIFAR10(root=path, train=True,
                                 download=True, transform=transform_train)

    val_set = datasets.CIFAR10(root=path, train=True,
                               download=True, transform=transform_val)

    test_set = datasets.CIFAR10(root=path, train=False,
                                download=True, transform=transform_test)

    indices = list(range(train_size + val_size))
    if shuffle:
        np.random.seed(1)  # Hardcoded seed for the moment
        np.random.shuffle(indices)
    train_idx, valid_idx = indices[:train_size], indices[train_size:]

    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                               sampler=train_sampler, num_workers=2)
    valid_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size,
                                               sampler=valid_sampler, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size,
                                              shuffle=False, num_workers=2)

    return train_loader, valid_loader, test_loader


if __name__ == "__main__":
    trainset, trainloader, testset, testloader = get_dataloaders()
    print(trainset.data.shape)
