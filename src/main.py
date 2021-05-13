"""
el main 8==D
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from d02_data.load_data import get_dataloaders_ssl
from d04_mixmatch.wideresnet import WideResNet
from mixmatch import MixMatch

if __name__ == '__main__':

    batch_size = 64
    num_labeled = 250
    n_steps = 10
    K = 2
    lambda_u = 75
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    labeled_loader, unlabeled_loader, val_loader, test_loader = get_dataloaders_ssl(path='../data',
                                                                                    batch_size=batch_size,
                                                                                    num_labeled=num_labeled)
    model = WideResNet(depth=28, k=2, n_out=10)
    model.to(device)

    criterion_x = nn.CrossEntropyLoss()
    criterion_u = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.002)

    iter_labeled_loader = iter(labeled_loader)
    iter_unlabeled_loader = iter(unlabeled_loader)

    for step in range(n_steps):

        # try-catch of dataloaders
        try:
            x_imgs, x_labels = iter_labeled_loader.next()
            # Check if batch size has been cropped for last batch
            if x_imgs.shape[0] < batch_size:
                iter_labeled_loader = iter(labeled_loader)
                x_imgs, x_labels = iter_labeled_loader.next()
        except:
            iter_loader = iter(labeled_loader)
            x_imgs, x_labels = iter_loader.next()

        try:
            u_imgs, _ = iter_unlabeled_loader.next()
            if u_imgs.shape[0] < batch_size:
                iter_unlabeled_loader = iter(unlabeled_loader)
                u_imgs, _ = iter_unlabeled_loader.next()
        except:
            iter_unlabeled_loader = iter(unlabeled_loader)
            u_imgs, _ = iter_unlabeled_loader.next()

        mixmatch = MixMatch(model, batch_size)
        x, u = mixmatch.run(x_imgs, x_labels, u_imgs)

        x_input = x[0]
        x_targets = x[1]
        u_input = u[0]
        u_targets = u[1].detach_()  # stop gradients from propagation to label guessing. Is this necessary?

        # Compute X' predictions
        x_output = model(x_input)

        # Compute U' predictions
        # Separate in batches (does it make any difference? maybe on batchnorm. Original implementation does it)
        u_batch_outs = []
        for k in range(K):
            u_batch = u_input[k * batch_size:(k + 1) * batch_size]
            u_batch_outs.append(model(u_batch))
        u_outputs = torch.cat(u_batch_outs, dim=0)

        # Compute loss
        lx = criterion_x(x_output, x_targets)
        lu = criterion_u(u_outputs, u_targets)
        loss = lx + lambda_u * lu  # TODO ramp up lambda_u

        # Step
        loss.backward()
        optimizer.step()
