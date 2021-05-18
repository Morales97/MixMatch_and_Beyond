import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from d02_data.load_data_idxs import get_dataloaders_with_index
from d04_mixmatch.wideresnet import WideResNet
from mixmatch import MixMatch


class PseudoLabelTrainer:

    def __init__(self, batch_size, num_lbls, model_params, n_steps, K, lambda_u, optimizer, adam,
                 sgd, steps_validation, steps_checkpoint, dataset, model_state_dict=None, ema_state_dict=None,
                 optim_state_dict=None):

        self.n_steps = n_steps
        self.start_step = 0
        self.K = K
        self.steps_validation = steps_validation
        self.steps_checkpoint = steps_checkpoint
        self.num_labeled = num_lbls
        self.labeled_loader, self.unlabeled_loader, self.val_loader, self.test_loader, self.lbl_idx, self.unlbl_idx, \
            self.val_idx = get_dataloaders_with_index(path='../data', batch_size=batch_size, num_labeled=num_lbls, which_dataset=dataset)
        self.batch_size = self.labeled_loader.batch_size
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(self.device)

        # Pseudo label
        self.steps_pseudo_lbl = 100
        self.tau = 0.95  # confidence threshold
        self.unlabeled_loader_original = self.unlabeled_loader

        depth, k, n_out = model_params
        self.model = WideResNet(depth=depth, k=k, n_out=n_out, bias=False).to(self.device)
        self.ema_model = WideResNet(depth=depth, k=k, n_out=n_out, bias=False).to(self.device)
        for param in self.ema_model.parameters():
            param.detach_()


        if optimizer == 'adam':
            self.lr, self.weight_decay = adam
            self.momentum, self.lr_decay = None, None
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
            self.ema_optimizer = WeightEMA(self.model, self.ema_model, self.lr, alpha=0.999)

        else:
            lr, momentum, weight_decay, lr_decay = sgd
            self.optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay,
                                       nesterov=True)
            self.learning_steps = lr_decay
            self.ema = None

        self.lambda_u_max, self.step_top_up = lambda_u
        self.loss_mixmatch = Loss(self.lambda_u_max, self.step_top_up)
        self.criterion = nn.CrossEntropyLoss()

        self.train_accuracies, self.train_losses, = [], []
        self.val_accuracies, self.val_losses, = [], []
        self.best_acc = 0

        self.mixmatch = MixMatch(self.model, self.batch_size, self.device)

        self.writer = SummaryWriter()

    def train(self):

        iter_labeled_loader = iter(self.labeled_loader)
        iter_unlabeled_loader = iter(self.unlabeled_loader)

        for step in range(self.start_step, self.n_steps):
            # Get next batch of data
            self.model.train()
            try:
                x_imgs, x_labels, _ = iter_labeled_loader.next()
                # Check if batch size has been cropped for last batch
                if x_imgs.shape[0] < self.batch_size:
                    iter_labeled_loader = iter(self.labeled_loader)
                    x_imgs, x_labels, _ = iter_labeled_loader.next()
            except:
                iter_labeled_loader = iter(self.labeled_loader)
                x_imgs, x_labels, _ = iter_labeled_loader.next()

            try:
                u_imgs, _, _ = iter_unlabeled_loader.next()
                if u_imgs.shape[0] < self.batch_size:
                    iter_unlabeled_loader = iter(self.unlabeled_loader)
                    u_imgs, _, _ = iter_unlabeled_loader.next()
            except:
                iter_unlabeled_loader = iter(self.unlabeled_loader)
                u_imgs, _, _ = iter_unlabeled_loader.next()

            # Send to GPU
            x_imgs = x_imgs.to(self.device)
            x_labels = x_labels.to(self.device)
            u_imgs = u_imgs.to(self.device)

            # MixMatch algorithm
            x, u = self.mixmatch.run(x_imgs, x_labels, u_imgs)
            x_input, x_targets = x
            u_input, u_targets = u
            u_targets.detach_()  # stop gradients from propagation to label guessing. Is this necessary?

            # Compute X' predictions
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
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.ema_optimizer.step()

            '''
            # Decaying learning rate. Used in with SGD Nesterov optimizer
            if not self.ema and step in self.learning_steps:
                for g in self.optimizer.param_groups:
                    g['lr'] *= 0.2
            '''

            # Evaluate model
            self.model.eval()
            if not step % self.steps_validation:
                val_acc, is_best = self.evaluate_loss_acc(step)
                if is_best:
                    self.save_model(step=step, path='../models/best_checkpoint.pt')

            # Save checkpoint
            if not step % self.steps_checkpoint:
                self.save_model(step=step, path='../models/checkpoint.pt')

            # Generate Pseudo-labels
            if not step % self.steps_pseudo_lbl:
                pseudo_labels, indices, unlbl_indices = self.get_pseudo_labels()

                # Check how many pseudo labels are correct
                if True:
                    true_labels = self.unlabeled_loader.dataset.targets
                    correct_pseudo_labels = 0
                    for i, index in enumerate(indices):
                        if pseudo_labels[i] == true_labels[index]:
                            correct_pseudo_labels += 1
                    print('*** %d pseudo labels generated, %d correspond to the ground truth' %
                          (pseudo_labels.shape[0], correct_pseudo_labels))

                # Update loaders
                new_lbl_idx = torch.cat((torch.tensor(self.lbl_idx, device=self.device), indices)).tolist()
                new_unlbl_idx = unlbl_indices.tolist()
                self.labeled_loader, self.unlabeled_loader, self.val_loader, self.test_loader, _, _, _ = \
                    get_dataloaders_with_index(path='../data',
                                               batch_size=self.batch_size,
                                               num_labeled=self.num_labeled,
                                               which_dataset='cifar10',
                                               lbl_idxs=new_lbl_idx,
                                               unlbl_idxs=new_unlbl_idx,
                                               valid_idxs=self.val_idx)
                iter_labeled_loader = iter(self.labeled_loader)
                iter_unlabeled_loader = iter(self.unlabeled_loader)

                print('Training with Labeled / Unlabeled / Validation samples\t %d %d %d' % (len(new_lbl_idx),
                      len(new_unlbl_idx), len(self.val_idx)))


        # --- Training finished ---
        test_val, test_acc = self.evaluate(self.test_loader)
        print("Training done!!\t Test loss: %.3f \t Test accuracy: %.3f" % (test_val, test_acc))

        self.writer.flush()

    # --- support functions ---

    def get_pseudo_labels(self):

        pseudo_labels_matrix = torch.tensor([]).to(self.device)
        new_unlbl_indxs = torch.tensor([]).to(self.device)
        for batch_idx, (data, target, idx) in enumerate(self.unlabeled_loader_original):
            with torch.no_grad():
                # Get predictions for unlabeled samples
                out = self.model(data.to(self.device))
                p_out = torch.softmax(out, dim=1)   # turn into probability distribution
                p_pseudo_lbl, pseudo_lbl = torch.max(p_out, dim=1)

                # Apply threshold and concat
                pseudo_lbl_matrix = torch.vstack((p_pseudo_lbl, pseudo_lbl, idx.to(self.device)))
                pseudo_lbl_matrix = pseudo_lbl_matrix[:, pseudo_lbl_matrix[0] >= self.tau]
                unlbl_indxs = pseudo_lbl_matrix[2, pseudo_lbl_matrix[0] < self.tau]

                new_unlbl_indxs = torch.cat((new_unlbl_indxs, unlbl_indxs))
                pseudo_labels_matrix = torch.cat((pseudo_labels_matrix, pseudo_lbl_matrix))

        pseudo_labels = pseudo_labels_matrix[1]
        indices = pseudo_labels_matrix[2]
        print(new_unlbl_indxs.shape)
        return pseudo_labels, indices, new_unlbl_indxs

    def evaluate_loss_acc(self, step):
        val_loss, val_acc = self.evaluate(self.val_loader)
        self.val_losses.append(val_loss)
        self.val_accuracies.append(val_acc)
        train_loss, train_acc = self.evaluate(self.labeled_loader)
        self.train_losses.append(train_loss)
        self.train_accuracies.append(train_acc)

        is_best = False
        if val_acc > self.best_acc:
            self.best_acc = val_acc
            is_best = True

        print("Step %d.\tLoss train_lbl/valid  %.2f  %.2f\t Accuracy train_lbl/valid  %.2f  %.2f \tBest acc %.2f \t%s" %
              (step, train_loss, val_loss, train_acc, val_acc, self.best_acc, time.ctime()))

        self.writer.add_scalar("Loss train_label", train_loss, step)
        self.writer.add_scalar("Loss validation", val_loss, step)
        self.writer.add_scalar("Accuracy train_label", train_acc, step)
        self.writer.add_scalar("Accuracy validation", val_acc, step)
        return val_acc, is_best

    def evaluate(self, dataloader):
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

    def save_model(self, step=None, path='../models/model.pt'):
        loss_list, lx, lu, lu_weighted = self.get_losses()
        if not step:
            step = self.n_steps     # Training finished

        torch.save({
            'step': step,
            'model_state_dict': self.model.state_dict(),
            'ema_state_dict': self.ema_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss_train': self.train_losses,
            'loss_val': self.val_losses,
            'acc_train': self.train_accuracies,
            'acc_val': self.val_accuracies,
            'loss_batch': loss_list,
            'lx': lx,
            'lu': lu,
            'lu_weighted': lu_weighted,
            'steps': self.n_steps,
            'batch_size': self.batch_size,
            'num_labels': self.num_labeled,
            'lambda_u_max': self.lambda_u_max,
            'step_top_up': self.step_top_up,
            'lr': self.lr,
            'weight_decay': self.weight_decay,
            'momentum': self.momentum,
            'lr_decay': self.lr_decay,
            'lbl_idx': self.lbl_idx,
            'unlbl_idx': self.unlbl_idx,
            'val_idx': self.val_idx,
        }, path)

    def load_checkpoint(self, model_name):
        saved_model = torch.load(f'../models/{model_name}')
        self.model.load_state_dict(saved_model['model_state_dict'])
        self.ema_model.load_state_dict(saved_model['ema_state_dict'])
        self.optimizer.load_state_dict(saved_model['optimizer_state_dict'])
        self.start_step = 40_000  # saved_model['step']
        # self.train_losses = saved_model['loss_train']
        # self.val_losses = saved_model['loss_val']
        # self.train_accuracies = saved_model['acc_train']
        # self.val_accuracies = saved_model['acc_val']
        print('Model ' + model_name + ' loaded.')


class Loss(object):

    def __init__(self, lambda_u_max, step_top_up):
        self.lambda_u_max = lambda_u_max
        self.step_top_up = step_top_up
        self.mse_loss = nn.MSELoss()
        self.lx_list = []
        self.lu_list = []
        self.lu_weighted_list = []
        self.loss_list = []

    def __call__(self, x_output, x_target, u_output, u_target, step):
        lambda_u = self.ramp_up_lambda(step)
        u_output = torch.softmax(u_output, dim=1)

        lx = - torch.mean(torch.sum(x_target * torch.log_softmax(x_output, dim=1), dim=1))
        # lx = torch.clamp(lx, min=0, max=2)  # Try clamping lx to
        lu = self.mse_loss(u_output, u_target)
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


class WeightEMA(object):
    def __init__(self, model, ema_model, lr, alpha=0.999):
        self.model = model
        self.ema_model = ema_model
        self.alpha = alpha
        self.params = list(model.state_dict().values())
        self.ema_params = list(ema_model.state_dict().values())
        self.wd = 0.02 * lr

        for param, ema_param in zip(self.params, self.ema_params):
            param.data.copy_(ema_param.data)

    def step(self):
        one_minus_alpha = 1.0 - self.alpha
        for param, ema_param in zip(self.params, self.ema_params):
            if ema_param.dtype == torch.float32:
                # Update Exponential Moving Average parameters
                ema_param.mul_(self.alpha)
                ema_param.add_(param * one_minus_alpha)
                # Apply Weight Decay
                param.mul_(1 - self.wd) # Beware that this "param" affects the main model. It is passed by reference
