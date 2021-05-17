from pathlib import Path
import yaml
import time
import datetime
import torch

from d07_visualization.viz_training import plot_acc, plot_training_loss, plot_losses
from mixmatch_trainer import MixMatchTrainer
from fs_tainer import FullySupervisedTrainer

if __name__ == '__main__':
    print("Starting main...")
    configuration = yaml.load(Path("config.yml").read_text(), Loader=yaml.SafeLoader)

    config = configuration['MixMatchTrainer']
    params = configuration['WideResNet']
    adam = config['adam']
    sgd = config['sgd']
    lambda_u = config['lambda_u']

    batch_size = configuration['cifar_10']['batch_size']
    num_labeled = configuration['cifar_10']['num_labeled']
    n_steps = config['n_steps']
    K = config['K']
    lambda_u_params = lambda_u['lambda_u_max'], lambda_u['step_top_up']
    steps_validation = config['steps_validation']
    steps_checkpoint = config['steps_checkpoint']
    optimizer = config['optimizer']
    adam_params = adam['lr'], adam['weight_decay']
    sgd_params = sgd['lr'], sgd['momentum'], sgd['weight_decay'], sgd['lr_decay_steps']

    wideresnet_params = (params['depth'], params['k'], params['n_out'])


    trainer = MixMatchTrainer(batch_size, num_labeled, wideresnet_params, n_steps, K, lambda_u_params,
                              optimizer, adam_params, sgd_params, steps_validation, steps_checkpoint)

    # trainer = FullySupervisedTrainer(batch_size, wideresnet_params, n_steps, optimizer, adam_params, sgd_params,
    #                                 steps_validation, steps_checkpoint)

    start_time = time.time()

    # trainer.load_checkpoint('250_lbl_40k_steps_bias_trans.pt')

    trainer.train()

    seconds = time.time() - start_time
    print("Time elapsed: " + str(datetime.timedelta(seconds=seconds)))

    trainer.save_model()

    # plot_training_loss(trainer.train_losses, trainer.val_losses)
    # plot_acc(trainer.train_accuracies, trainer.val_accuracies)
    # plot_losses(*trainer.get_losses())
