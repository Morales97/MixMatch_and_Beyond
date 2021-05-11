import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pdb

from d01_utils.torch_ema import ExponentialMovingAverage
from d02_data.load_data import get_dataloaders, get_dataloaders_validation
from d03_processing.transform_data import Augment
from d04_mixmatch.wideresnet import WideResNet
from d07_visualization.visualize_cifar10 import show_img


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Assuming that we are on a CUDA machine, this should print a CUDA device:
    print(device)

    # trainset, train_loader, testset, test_loader = get_dataloaders(path='../data')
    train_loader, val_loader, test_loader = get_dataloaders_validation(path='../data', batch_size=64)
    model = WideResNet(depth=28, k=2, n_out=10)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=4e-4)
    ema = ExponentialMovingAverage(model.parameters(), decay=0.999)

    # show_img(iter(trainloader).next()[0][0])

    n_batches_train = np.floor(train_loader.sampler.__len__() / train_loader.batch_size)
    n_batches_val = np.floor(val_loader.sampler.__len__() / val_loader.batch_size)
    loss_train, loss_val = [], []

    augment = Augment(K=1)

    for epoch in range(2):  # loop over the dataset multiple times

        model.train()     # set to train mode
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            # inputs, labels = data
            inputs, labels = data[0], data[1].to(device)  # send to cuda
            inputs = augment(inputs)[0]
            inputs = inputs.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            ema.update(model.parameters())

            if i % 150 == 0:    # print every 300 mini-batches
                print('Step ' + str(i))


        # Compute train and validation loss
        model.eval()    # set to evaluation mode. important for batchnorm layer
        with torch.no_grad():
            train_loss = 0
            for i, data in enumerate(train_loader, 0):
                inputs, labels = data[0].to(device), data[1].to(device)
                outputs = model(inputs)
                train_loss += criterion(outputs, labels).item()
            train_loss /= n_batches_train
            loss_train.append(train_loss)
            val_loss = 0
            for i, data in enumerate(val_loader, 0):
                inputs, labels = data[0].to(device), data[1].to(device)
                outputs = model(inputs)
                val_loss += criterion(outputs, labels).item()
            val_loss /= n_batches_val
            loss_val.append(val_loss)

        print("Epoch %d done.\t Train loss: %.3f \t Validation loss: %.3f" %
              (epoch+1, train_loss, val_loss))

    print('Finished Training')


    # calculate accuracy on test set
    correct, total = 0, 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    model.eval()
    with torch.no_grad():
        for data in test_loader:
            images, labels = data[0].to(device), data[1].to(device)
            # calculate outputs by running images through the network
            outputs = model(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (
            100 * correct / total))


    # with EMA
    # First save original parameters before replacing with EMA version
    ema.store(model.parameters())
    # Copy EMA parameters to model
    ema.copy_to(model.parameters())
    model.eval()
    with torch.no_grad():
        for data in test_loader:
            images, labels = data[0].to(device), data[1].to(device)
            # calculate outputs by running images through the network
            outputs = model(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy tested with EMA parameters on the 10000 test images: %d %%' % (
            100 * correct / total))
    # Restore original parameters to resume training later
    ema.restore(model.parameters())
