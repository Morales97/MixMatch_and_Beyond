from pathlib import Path
import yaml
import time

from d02_data.load_data import get_dataloaders_ssl
from d07_visualization.viz_training import plot_acc, plot_training_loss, plot_losses
from mixmatch_trainer import MixMatchTrainer

if __name__ == '__main__':
	print("Starting main...")
	configuration = yaml.load(Path("config.yml").read_text(), Loader=yaml.SafeLoader)

	config = configuration['MixMatchTrainer']
	params = configuration['WideResNet']
	adam = config['adam']
	sgd = config['sgd']

	batch_size = configuration['cifar_10']['batch_size']
	num_labeled = configuration['cifar_10']['num_labeled']
	n_steps = config['n_steps']
	K = config['K']
	lambda_u_max = config['lambda_u_max']
	steps_validation = config['steps_validation']
	steps_checkpoint = config['steps_checkpoint']
	step_top_up = config['step_top_up']
	optimizer = config['optimizer']
	adam_params = adam['lr'], adam['weight_decay']
	sgd_params = sgd['lr'], sgd['momentum'], sgd['weight_decay'], sgd['lr_decay_steps']

	wideresnet_params = (params['depth'], params['k'], params['n_out'])

	data = get_dataloaders_ssl(path='../data', batch_size=batch_size, num_labeled=num_labeled)

	trainer = MixMatchTrainer(data, wideresnet_params, n_steps, K, lambda_u_max, steps_validation, step_top_up,
	                          optimizer, adam_params, sgd_params, steps_checkpoint)
	start_time = time.time()

	trainer.train()

	print("--- %s seconds ---" % (time.time() - start_time))

	trainer.save_model()

	plot_training_loss(trainer.train_losses, trainer.val_losses)
	plot_acc(trainer.train_accuracies, trainer.val_accuracies)
	plot_losses(*trainer.get_losses())
