import torchvision.transforms as transforms
import torch


class Augment:
    def __init__(self, K=2):
        self.K = K
        self.transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                            transforms.RandomAffine(degrees = 0, translate = (0.1, 0.1)),
                                            AddGaussianNoise(mean=0, std=0.15)])

    def __call__(self, batch):
        augmented_batch = torch.zeros((self.K, *batch.shape))
        for k in range(self.K):
            augmented_batch[k] = self.transform(batch)
        return augmented_batch


class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)