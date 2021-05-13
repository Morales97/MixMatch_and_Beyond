import torch
import torch.nn as nn
import torch.optim as optim

from d02_data.load_data import get_dataloaders_ssl
from d04_mixmatch.wideresnet import WideResNet
from mixmatch import MixMatch
from tqdm import tqdm


class Loss(object):

	def __init__(self, lambda_u_max, step_top_up):
		self.lambda_u_max = lambda_u_max
		self.step_top_up = step_top_up
		self.lx_list = []
		self.lu_list = []
		self.losses = []

	def __call__(self, x_output, x_target, u_output, u_target, step):

		lambda_u = self.ramp_up_lambda(step)

		lx = - torch.mean(torch.sum(x_target * torch.log_softmax(x_output, dim=1), dim=1))
		mse_loss = nn.MSELoss()
		output = torch.softmax(u_output, dim=1)
		lu = mse_loss(output, u_target) / u_target.shape[1]
		self.lu_list.append(lu)
		self.lx_list.append(lx)
		loss = lx + lu * lambda_u
		self.losses.append(loss)
		return loss

	def ramp_up_lambda(self, step):
		if step > self.step_top_up:
			return self.lambda_u_max
		else:
			return self.lambda_u_max * step / self.step_top_up


class MixMatchTrainer:

	def __init__(self, data, model_params, n_steps, K, lambda_u_max, steps_validation, lr, step_top_up):

		self.n_steps = n_steps
		self.K = K
		self.lambda_u_max = lambda_u_max
		self.steps_validation = steps_validation

		self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

		depth, k, n_out = model_params
		self.model = WideResNet(depth=depth, k=k, n_out=n_out)

		self.labeled_loader, self.unlabeled_loader, self.val_loader, self.test_loader = data
		self.batch_size = self.labeled_loader.batch_size

		self.val_accuracies = []
		self.val_losses = []

		self.criterion_x = nn.CrossEntropyLoss()
		self.criterion_u = nn.MSELoss()
		self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
		self.loss_mixmatch = Loss(self.lambda_u_max, step_top_up)
		self.criterion = nn.CrossEntropyLoss()

	def train(self):

		iter_labeled_loader = iter(self.labeled_loader)
		iter_unlabeled_loader = iter(self.unlabeled_loader)

		for step in range(self.n_steps):
			self.model.train()
			# try-catch of dataloaders
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

			self.model.to('cpu')
			mixmatch = MixMatch(self.model, self.batch_size)  # try if model is a reference, take this out of the steps loop
			x, u = mixmatch.run(x_imgs, x_labels, u_imgs)

			x_input, x_targets = x
			u_input, u_targets = u
			u_targets.detach_()  # stop gradients from propagation to label guessing. Is this necessary?

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

			if not step % self.steps_validation:
				# Compute train and validation loss
				val_loss, val_acc = self.evaluate(self.val_loader)
				self.val_accuracies.append(val_acc)
				self.val_losses.append(val_loss)
				print("Epoch %d done.\t Validation loss: %.3f \t Validation accuracy: %.3f" %
				      (step, val_loss, val_acc))

	def evaluate(self, dataset):
		self.model.eval()
		correct, total = 0, 0
		with torch.no_grad():
			val_loss = 0
			for i, data in enumerate(dataset, 0):
				inputs, labels = data[0].to(self.device), data[1].to(self.device)
				outputs = self.model(inputs)
				val_loss += self.criterion(outputs, labels).item()

				_, predicted = torch.max(outputs.data, 1)
				total += labels.size(0)
				correct += (predicted == labels).sum().item()
			val_loss /= dataset.__len__()

		val_acc = correct / total
		return val_loss, val_acc

	def get_loss(self):
		return self.val_losses, self.loss_mixmatch.losses, self.loss_mixmatch.lx_list, self.loss_mixmatch.lu_list
