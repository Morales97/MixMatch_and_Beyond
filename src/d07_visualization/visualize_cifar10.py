import matplotlib.pyplot as plt
import numpy as np

def show_img(img):
    img = img / 5 + 0.47     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()