import torchvision.transforms as transforms
import torch
import numpy as np


class Augment:
    def __init__(self, K=2):
        self.K = K
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.transform = transforms.Compose([transforms.RandomHorizontalFlip().to(self.device),
                                             RandomCrop(max_crop=4)])
                                            # transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)).to(self.device)])
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


class RandomCrop(torch.nn.Module):
    """
    1- Pad an image with reflection a random number of pixels in each side
    2- Crop the same number of pixels, in random side
    Returns an image of the same input size
    """
    def __init__(self, max_crop):
        super().__init__()
        self.max_crop = max_crop

    def forward(self, img):
        h, w = img.shape[1:]
        crop_size = np.random.randint(1, self.max_crop)
        img = np.pad(img, [(0, 0), (crop_size, crop_size), (crop_size, crop_size)], mode='reflect')

        top = np.random.randint(0, crop_size)
        left = np.random.randint(0, crop_size)

        img = img[:, top: top + h, left: left + w]
        return torch.from_numpy(img)



