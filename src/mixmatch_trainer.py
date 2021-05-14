import torch
import torch.nn as nn
import torch.optim as optim

from d02_data.load_data import get_dataloaders_ssl
from d04_mixmatch.wideresnet import WideResNet
from mixmatch import MixMatch
from tqdm import tqdm


class MixMatchTrainer:

    def __init__(self, data, model_params, n_steps, K, lambda_u_max, steps_validation, step_top_up, optimizer, adam,
                 sgd):

        self.n_steps = n_steps
        self.K = K
        self.lambda_u_max = lambda_u_max
        self.steps_validation = steps_validation
        self.labeled_loader, self.unlabeled_loader, self.val_loader, self.test_loader = data
        self.batch_size = self.labeled_loader.batch_size

        depth, k, n_out = model_params
        self.model = WideResNet(depth=depth, k=k, n_out=n_out)
        if optimizer == 'adam':
            lr, weight_decay = adam
            self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            lr, momentum, weight_decay = sgd
            self.optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

        self.loss_mixmatch = Loss(self.lambda_u_max, step_top_up)
        self.criterion = nn.CrossEntropyLoss()

        self.train_accuracies, self.train_losses = [], []
        self.val_accuracies, self.val_losses = [], []

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def train(self):

        iter_labeled_loader = iter(self.labeled_loader)
        iter_unlabeled_loader = iter(self.unlabeled_loader)

        for step in range(self.n_steps):
            # Get next batch of data
            try:
                x_imgs, x_labels = iter_labeled_loader.next()
                # Check if batch size has been cropped for last batch
                if x_imgs.shape[0] < self.batch_size:
                    iter_labeled_loader = iter(self.labeled_loader)
                    x_imgs, x_labels = iter_labeled_loader.next()
            except:
                iter_loader = iter(self.labeled_loader)
                x_imgs, x_labels = iter_loader.next()

            try:
                u_imgs, _ = iter_unlabeled_loader.next()
                if u_imgs.shape[0] < self.batch_size:
                    iter_unlabeled_loader = iter(self.unlabeled_loader)
                    u_imgs, _ = iter_unlabeled_loader.next()
            except:
                iter_unlabeled_loader = iter(self.unlabeled_loader)
                u_imgs, _ = iter_unlabeled_loader.next()

            # MixMatch algorithm
            self.model.to('cpu')
            mixmatch = MixMatch(self.model,
                                self.batch_size)  # try if model is a reference, take this out of the steps loop
            x, u = mixmatch.run(x_imgs, x_labels, u_imgs)

            x_input, x_targets = x
            u_input, u_targets = u
            u_targets.detach_()  # stop gradients from propagation to label guessing. Is this necessary?

            # Send to GPU
            self.model.train()
            self.model.to(self.device)
            x_input = x_input.to(self.device)
            x_targets = x_targets.to(self.device)
            u_input = u_input.to(self.device)
            u_targets = u_targets.to(self.device)

            # Compute X' predictions
            self.optimizer.zero_grad()
            x_output = self.model(x_input)

            # Compute U' predictions
            # Separate in batches (does it make any difference? maybe on batchnorm. Original implementation does it)
            u_batch_outs = []
            for k in range(self.K):
                u_batch = u_input[k * self.batch_size:(k + 1) * self.batch_size]
                u_batch_outs.append(self.model(u_batch))
            u_outputs = torch.cat(u_batch_outs, dim=0)

            # Compute loss
            loss = self.loss_mixmatch(x_output, x_targets, u_outputs, u_targets, step)

            # Step
            loss.backward()
            self.optimizer.step()

            # Evaluate model
            if not step % self.steps_validation:
                val_loss, val_acc = self.evaluate(self.val_loader)
                self.val_losses.append(val_loss)
                self.val_accuracies.append(val_acc)
                train_loss, train_acc = self.evaluate(self.labeled_loader)
                self.train_losses.append(train_loss)
                self.train_accuracies.append(train_acc)
                print("Step %d.\t Loss train_lbl/valid  %.2f  %.2f \t Accuracy train_lbl/valid  %.2f  %.2f" %
                      (step, train_loss, val_loss, train_acc, val_acc))

        test_val, test_acc = self.evaluate(self.test_loader)
        print("Training done!!\t Test loss: %.3f \t Test accuracy: %.3f" % (test_val, test_acc))

    def evaluate(self, dataloader):
        self.model.eval()
        correct, total, loss = 0, 0, 0
        with torch.no_grad():
            for i, data in enumerate(dataloader, 0):
                inputs, labels = data[0].to(self.device), data[1].to(self.device)
                outputs = self.model(inputs)
                loss += self.criterion(outputs, labels).item()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            loss /= dataloader.__len__()

        acc = correct / total * 100
        return loss, acc

    def get_losses(self):
        return self.loss_mixmatch.loss_list, self.loss_mixmatch.lx_list, self.loss_mixmatch.lu_list, self.loss_mixmatch.lu_weighted_list


class Loss(object):

    def __init__(self, lambda_u_max, step_top_up):
        self.lambda_u_max = lambda_u_max
        self.step_top_up = step_top_up
        self.lx_list = []
        self.lu_list = []
        self.lu_weighted_list = []
        self.loss_list = []

    def __call__(self, x_output, x_target, u_output, u_target, step):
        lambda_u = self.ramp_up_lambda(step)
        mse_loss = nn.MSELoss()
        u_output = torch.softmax(u_output, dim=1)

        lx = - torch.mean(torch.sum(x_target * torch.log_softmax(x_output, dim=1), dim=1))
        lu = mse_loss(u_output, u_target) / u_target.shape[1]
        loss = lx + lu * lambda_u

        self.lx_list.append(lx)
        self.lu_list.append(lu)
        self.lu_weighted_list.append(lu * lambda_u)
        self.loss_list.append(loss)
        return loss

    def ramp_up_lambda(self, step):
        if step > self.step_top_up:
            return self.lambda_u_max
        else:
            return self.lambda_u_max * step / self.step_top_up