import torch
import torchvision
import torchvision.transforms as transforms

def get_dataloaders( path='../../data' ):

    transform_train = transforms.Compose(
        # transforms.RandomCrop(32, padding=4),
        # transforms.RandomHorizontalFlip(),
        [transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    batch_size = 4

    trainset = torchvision.datasets.CIFAR10(root=path, train=True,
                                            download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root=path, train=False,
                                        download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    return trainset, trainloader, testset, testloader

if __name__ == "__main__":
    trainset, trainloader, testset, testloader = get_dataloaders()
    print(trainset.data.shape)