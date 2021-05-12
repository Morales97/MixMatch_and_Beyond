
import os
import sys

root_dir = os.path.dirname(os.getcwd())
sys.path.append(root_dir)
print(root_dir)

from d03_processing.transform_data import Augment


class MixMatch(object):

    def __init__(self, T=0.5, K=2, alpha=0.75):
        self.T = T
        self.K = K
        self.alpha = alpha

    def mixmatch_algorithm(self, x_imgs, x_labels, u_imgs):
        augment_once = Augment(K=1)
        augment_k = Augment(K=self.K)

        x_hat = augment_once(x_imgs)
        u_hat = augment_k(u_imgs)

        return x_hat, u_hat


if __name__ == '__main__':
    print('foo')