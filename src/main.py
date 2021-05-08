"""
el main 8==D
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pdb

from d02_data.load_data import get_dataloaders, get_dataloaders_validation
from d03_processing.transform_data import TransformTwice
from d04_mixmatch.vanilla_net import VanillaNet
from d07_visualization.visualize_cifar10 import show_img

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Assuming that we are on a CUDA machine, this should print a CUDA device:
    print(device)

    augment_labeled = TransformTwice(K=1)
    augment_unlabeled = TransformTwice(K=2)

    # trainset, train_loader, testset, test_loader = get_dataloaders(path='../data')
    train_loader, val_loader, test_loader = get_dataloaders_validation(path='../data', batch_size=16)
    net = VanillaNet()
    net.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # show_img(iter(trainloader).next()[0][0])

    n_batches_train = np.floor(train_loader.sampler.__len__() / train_loader.batch_size)
    n_batches_val = np.floor(val_loader.sampler.__len__() / val_loader.batch_size)
    loss_train, loss_val = [], []

    for epoch in range(2):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            # inputs, labels = data
            inputs, labels = data[0].to(device), data[1].to(device)  # send to cuda

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            """
            running_loss += loss.item()
            if i % 300 == 99:    # print every 300 mini-batches
                print('[Epoch %d, Step %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 300))
                running_loss = 0.0
            """

        # Compute train and validation loss
        with torch.no_grad():
            train_loss = 0
            for i, data in enumerate(train_loader, 0):
                inputs, labels = data[0].to(device), data[1].to(device)
                outputs = net(inputs)
                train_loss += criterion(outputs, labels).item()
            train_loss /= n_batches_train
            loss_train.append(train_loss)
            val_loss = 0
            for i, data in enumerate(val_loader, 0):
                inputs, labels = data[0].to(device), data[1].to(device)
                outputs = net(inputs)
                val_loss += criterion(outputs, labels).item()
            val_loss /= n_batches_val
            loss_val.append(val_loss)

        print("Epoch %d done.\t Train loss: %.3f \t Validation loss: %.3f" %
              (epoch+1, train_loss, val_loss))

    print('Finished Training')


    # calculate accuracy on test set
    correct, total = 0, 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            # calculate outputs by running images through the network
            outputs = net(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (
            100 * correct / total))



