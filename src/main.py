from pathlib import Path
import yaml

from d02_data.load_data import get_dataloaders_ssl
from mixmatch_trainer import MixMatchTrainer

if __name__ == '__main__':
	configuration = yaml.load(Path("config.yml").read_text(), Loader=yaml.SafeLoader)

	config = configuration['MixMatchTrainer']
	params = configuration['WideResNet']

	batch_size = configuration['cifar_10']['batch_size']
	num_labeled = configuration['cifar_10']['num_labeled']
	n_steps = config['n_steps']
	K = config['K']
	lambda_u_max = config['lambda_u_max']
	steps_validation = config['steps_validation']
	lr = config['lr']
	step_top_up = config['step_top_up']

	wideresnet_params = (params['depth'], params['k'], params['n_out'])

	data = get_dataloaders_ssl(path='../data', batch_size=batch_size, num_labeled=num_labeled)

	trainer = MixMatchTrainer(data, wideresnet_params, n_steps, K, lambda_u_max, steps_validation, lr, step_top_up)

	trainer.train()
