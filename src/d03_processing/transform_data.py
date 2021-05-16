import torchvision.transforms as transforms
import torch


class Augment:
    def __init__(self, K=2):
        self.K = K
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.transform = transforms.Compose([transforms.RandomHorizontalFlip().to(self.device),
                                            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)).to(self.device)])
                                            # AddGaussianNoise(mean=0, std=0.15).to(self.device)]

    def __call__(self, batch):
        augmented_batch = torch.zeros((self.K, *batch.shape), device=self.device)
        for k in range(self.K):
            augmented_batch[k] = self.transform(batch)
        return augmented_batch


class AddGaussianNoise(torch.nn.Module):
    def __init__(self, mean=0., std=1.):
        super().__init__()
        self.std = std
        self.mean = mean
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
    def forward(self, tensor):
        return tensor + torch.randn(tensor.size(), device=self.device) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)