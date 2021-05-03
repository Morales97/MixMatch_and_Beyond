"""
el main 8==D
"""

import torch.nn as nn
import torch.optim as optim

from d02_data.load_data import get_dataloaders
from d04_mixmatch.vanilla_net import VanillaNet
from d07_visualization.visualize_cifar10 import show_img

if __name__ == '__main__':
    trainset, trainloader, testset, testloader = get_dataloaders(path='../data')
    net = VanillaNet()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    show_img(iter(trainloader).next()[0][0])

    for epoch in range(2):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    print('Finished Training')