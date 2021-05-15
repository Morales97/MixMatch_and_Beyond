import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from d01_utils.torch_ema import ExponentialMovingAverage
from d04_mixmatch.wideresnet import WideResNet
from mixmatch import MixMatch

class FullySupervisedTrainer:

    def __init__(self, data, model_params, n_steps, steps_validation, optimizer, adam,
                 sgd, steps_checkpoint):