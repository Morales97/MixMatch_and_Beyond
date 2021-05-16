import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from d01_utils.torch_ema import ExponentialMovingAverage
from d02_data.load_data import get_dataloaders_ssl
from d04_mixmatch.wideresnet import WideResNet


class FullySupervisedTrainer:

    def __init__(self, batch_size, model_params, n_steps, optimizer, adam,
                 sgd, steps_validation, steps_checkpoint):

        self.n_steps = n_steps
        self.steps_validation = steps_validation
        self.steps_checkpoint = steps_checkpoint
        self.train_loader, _, self.val_loader, self.test_loader = get_dataloaders_ssl\
            (path='../data', batch_size=batch_size, num_labeled=45000)
        self.batch_size = batch_size

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(self.device)

        depth, k, n_out = model_params
        self.model = WideResNet(depth=depth, k=k, n_out=n_out).to(self.device)
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

        self.criterion = nn.CrossEntropyLoss()

        self.train_accuracies, self.train_losses = [], []
        self.val_accuracies, self.val_losses = [], []

        self.writer = SummaryWriter()


    def train(self):

        iter_train_loader = iter(self.train_loader)

        for step in range(self.n_steps):
            # Get next batch of data

            try:
                x_input, x_labels = iter_train_loader.next()
                # Check if batch size has been cropped for last batch
                if x_input.shape[0] < self.batch_size:
                    iter_train_loader = iter(self.train_loader)
                    x_input, x_labels = iter_train_loader.next()
            except:
                iter_train_loader = iter(self.train_loader)
                x_input, x_labels = iter_train_loader.next()

            # Send to GPU
            x_input = x_input.to(self.device)
            x_labels = x_labels.to(self.device)

            # Compute X' predictions
            self.model.train()
            self.optimizer.zero_grad()
            x_output = self.model(x_input)

            # Compute loss
            loss = self.criterion(x_output, x_labels)

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
        train_loss, train_acc = self.evaluate(self.train_loader)
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
        train_loss, train_acc = self.evaluate(self.train_loader)
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

    def save_model(self):
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
        }, '../models/model.pt')
