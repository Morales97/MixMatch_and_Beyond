import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np

from d01_utils.torch_ema import ExponentialMovingAverage
from d02_data.load_data import get_dataloaders_ssl
from d03_processing.transform_data import Augment
from d04_mixmatch.wideresnet import WideResNet
from mixmatch import MixMatch


class MixMatchTrainerSelfContained:
    "No MixMatch object"

    def __init__(self, batch_size, num_lbls, model_params, n_steps, K, lambda_u, optimizer, adam,
                 sgd, steps_validation, steps_checkpoint):

        self.n_steps = n_steps
        self.K = K
        self.T = 2
        self.softmax = nn.Softmax(dim=1)
        self.steps_validation = steps_validation
        self.steps_checkpoint = steps_checkpoint
        self.labeled_loader, self.unlabeled_loader, self.val_loader, self.test_loader = get_dataloaders_ssl\
            (path='../data', batch_size=batch_size, num_labeled=num_lbls)
        self.batch_size = self.labeled_loader.batch_size
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(self.device)

        depth, k, self.n_out = model_params
        self.model = WideResNet(depth=depth, k=k, n_out=self.n_out).to(self.device)
        if optimizer == 'adam':
            lr, weight_decay = adam
            self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
            self.ema = ExponentialMovingAverage(self.model.parameters(), decay=0.999)

        else:
            lr, momentum, weight_decay, lr_decay = sgd
            self.optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay,
                                       nesterov=True)
            self.learning_steps = lr_decay
            self.ema = None

        self.lambda_u_max, self.step_top_up = lambda_u
        self.loss_mixmatch = Loss(self.lambda_u_max, self.step_top_up)
        self.criterion = nn.CrossEntropyLoss()

        self.train_accuracies, self.train_losses = [], []
        self.val_accuracies, self.val_losses = [], []

        self.writer = SummaryWriter()

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
                iter_labeled_loader = iter(self.labeled_loader)
                x_imgs, x_labels = iter_labeled_loader.next()

            try:
                u_imgs, _ = iter_unlabeled_loader.next()
                if u_imgs.shape[0] < self.batch_size:
                    iter_unlabeled_loader = iter(self.unlabeled_loader)
                    u_imgs, _ = iter_unlabeled_loader.next()
            except:
                iter_unlabeled_loader = iter(self.unlabeled_loader)
                u_imgs, _ = iter_unlabeled_loader.next()

            # Send to GPU
            x_imgs = x_imgs.to(self.device)
            x_labels = x_labels.to(self.device)
            u_imgs = u_imgs.to(self.device)

            x_labels = self.one_hot_encoding(x_labels)

            # Augment
            augment_once = Augment(K=1)
            augment_k = Augment(K=self.K)
            x_hat = augment_once(x_imgs)  # shape (1, batch_size, 3, 32, 32)
            u_hat = augment_k(u_imgs)  # shape (K, batch_size, 3, 32, 32)

            # Generate guessed labels
            q_bar = self.guess_label(u_hat)
            q = self.sharpen(q_bar)  # shape (K, batch_size, 10)

            x_hat = x_hat.reshape((-1, 3, 32, 32))  # shape (batch_size, 3, 32, 32)
            u_hat = u_hat.reshape((-1, 3, 32, 32))  # shape (K*batch_size, 3, 32, 32)
            q = q.repeat(self.K, 1, 1).reshape(-1, 10)  # shape (K*batch_size, 10)

            # Concat and shuffle
            w_imgs = torch.cat((x_hat, u_hat))
            w_labels = torch.cat((x_labels, q))
            w_imgs, w_labels = self.shuffle_matrices(w_imgs, w_labels)

            # Apply MixUp
            x_prime, p_prime = self.mixup(x_hat, w_imgs[:self.batch_size], x_labels, w_labels[:self.batch_size])
            u_prime, q_prime = self.mixup(u_hat, w_imgs[self.batch_size:], q, w_labels[self.batch_size:])






            # MixMatch algorithm
            x, u = self.mixmatch.run(x_imgs, x_labels, u_imgs)
            x_input, x_targets = x
            u_input, u_targets = u
            u_targets.detach_()  # stop gradients from propagation to label guessing. Is this necessary?

            # Compute X' predictions
            self.model.train()
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
            if self.ema: self.ema.update(self.model.parameters())

            # Decaying learning rate. Used in with SGD Nesterov optimizer
            if not self.ema and step in self.learning_steps:
                for g in self.optimizer.param_groups:
                    g['lr'] *= 0.2

            # Evaluate model
            if not step % self.steps_validation:
                self.evaluate_no_ema(step)

            # Evaluate with EMA
            if self.ema and not step % self.steps_validation:
                self.evaluate_ema(step)

            # Save checkpoint
            if not step % self.steps_checkpoint:
                torch.save({
                    'step': step,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                }, '../models/checkpoint.pt')

        # --- Training finished ---
        # Evaluate on test set
        test_val, test_acc = self.evaluate(self.test_loader)
        print("Training done!!\t Test loss: %.3f \t Test accuracy: %.3f" % (test_val, test_acc))

        # Evaluate with EMA
        if self.ema:
            self.ema.store(self.model.parameters())
            self.ema.copy_to(self.model.parameters())
            test_val, test_acc = self.evaluate(self.test_loader)
            print("With EMA\t Test loss: %.3f \t Test accuracy: %.3f" % (test_val, test_acc))
            self.ema.restore(self.model.parameters())

        self.writer.flush()

    # --- support functions ---

    def evaluate_no_ema(self, step):
        val_loss, val_acc = self.evaluate(self.val_loader)
        self.val_losses.append(val_loss)
        self.val_accuracies.append(val_acc)
        train_loss, train_acc = self.evaluate(self.labeled_loader)
        self.train_losses.append(train_loss)
        self.train_accuracies.append(train_acc)
        print("Step %d.\t Loss train_lbl/valid  %.2f  %.2f \t Accuracy train_lbl/valid  %.2f  %.2f \t %s" %
              (step, train_loss, val_loss, train_acc, val_acc, time.ctime()))

        self.writer.add_scalar("Loss train_label", train_loss, step)
        self.writer.add_scalar("Loss validation", val_loss, step)
        self.writer.add_scalar("Accuracy train_label", train_acc, step)
        self.writer.add_scalar("Accuracy validation", val_acc, step)

    def evaluate_ema(self, step):
        # First save original parameters before replacing with EMA version
        self.ema.store(self.model.parameters())
        # Copy EMA parameters to model
        self.ema.copy_to(self.model.parameters())
        val_loss, val_acc = self.evaluate(self.val_loader)
        train_loss, train_acc = self.evaluate(self.labeled_loader)
        print("With EMA.\t Loss train_lbl/valid  %.2f  %.2f \t Accuracy train_lbl/valid  %.2f  %.2f" %
              (train_loss, val_loss, train_acc, val_acc))
        self.ema.restore(self.model.parameters())

        self.writer.add_scalar("Loss train_label EMA", train_loss, step)
        self.writer.add_scalar("Loss validation EMA", val_loss, step)
        self.writer.add_scalar("Accuracy train_label EMA", train_acc, step)
        self.writer.add_scalar("Accuracy validation EMA", val_acc, step)

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

    def save_model(self):
        loss_list, lx, lu, lu_weighted = self.get_losses()
        model_state_dict = self.model.state_dict()
        ema_state_dict = None
        if self.ema:
            self.ema.copy_to(self.model.parameters())
            ema_state_dict = self.model.state_dict()

        torch.save({
            'model_state_dict': model_state_dict,
            'ema_state_dict': ema_state_dict,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss_train': self.train_losses,
            'loss_val': self.val_losses,
            'acc_train': self.train_accuracies,
            'acc_val': self.val_accuracies,
            'loss_batch': loss_list,
            'lx': lx,
            'lu': lu,
            'lu_weighted': lu_weighted,
        }, '../models/model.pt')

    def mixup(self, x1, x2, p1, p2):
        n_samples = x1.shape[0]
        lambda_rand = self.beta.sample([n_samples, 1, 1, 1]).to(self.device)  # one lambda per sample
        lambda_prime = torch.max(lambda_rand, 1 - lambda_rand).to(self.device)
        x_prime = lambda_prime * x1 + (1 - lambda_prime) * x2
        lambda_prime = lambda_prime.reshape(-1, 1)
        p_prime = lambda_prime * p1 + (1 - lambda_prime) * p2
        return x_prime, p_prime

    def sharpen(self, q_bar):
        q = torch.pow(q_bar, 1 / self.T) / torch.sum(torch.pow(q_bar, 1 / self.T), axis=1)[:, np.newaxis]
        return q

    def guess_label(self, u_hat):
        self.model.eval()
        with torch.no_grad():
            q_bar = torch.zeros([self.batch_size, self.n_out], device=self.device)
            for k in range(self.K):
                q_bar += self.softmax(self.model(u_hat[k]))
            q_bar /= self.K

        self.model.train()
        return q_bar

    def one_hot_encoding(self, labels):
        shape = (labels.shape[0], self.n_labels)
        one_hot = torch.zeros(shape, dtype=torch.float32, device=self.device)
        rows = torch.arange(labels.shape[0])
        one_hot[rows, labels] = 1
        return one_hot

    # shuffles along the first axis (axis 0)
    def shuffle_matrices(self, m1, m2):
        n_samples = m1.shape[0]
        rand_indexes = torch.randperm(n_samples)
        m1 = m1[rand_indexes]
        m2 = m2[rand_indexes]
        return m1, m2


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
        lu = mse_loss(u_output, u_target)
        loss = lx + lu * lambda_u

        self.lx_list.append(lx.item())
        self.lu_list.append(lu.item())
        self.lu_weighted_list.append(lu.item() * lambda_u)
        self.loss_list.append(loss.item())
        return loss

    def ramp_up_lambda(self, step):
        if step > self.step_top_up:
            return self.lambda_u_max
        else:
            return self.lambda_u_max * step / self.step_top_up