# MixMatch

PyTorch implementation of the MixMatch algorithm.

## Papers

- [MixMatch: A Holistic Approach to Semi-Supervised Learning](https://arxiv.org/abs/1905.02249)
<!---- [Unsupervised Representation Learning by Predicting Image Rotations](https://arxiv.org/abs/1803.07728) -->

## Environment creation

```
$ conda env create --file environment.yml
```
Change the desired parameters in ```config.yml``` and run:

```
$ python3 main.py
```
## Datasets
- [CIFAR-10](http://www.cs.toronto.edu/~kriz/cifar.html): The dataset gets downloaded automatically in ```data/cifar-10-batches-py```, if it has not been downloaded before.


## Results 
|  | 250 Labels | 1000 labels| 4000 labels| Fully supervised |
|:---|:---:|:---:|:---:|:---:|
|This code | 86.52 | 90.28 | 93.33 | 94.39 |
|MixMatch paper | 88.92 ± 0.87 | 92.25 ± 0.32| 93.76 ± 0.06|95.87|

# Beyond MixMatch

In this project, we attempt to improve MixMatch using _pseudo labels_. The basic idea is to use the most confident predictions by a model trained with MixMatch as one-hot labels. This allows unlabeled images to be inlcuded in the labeled dataset. 

## Results

We don't see a significant improvement in the use of _pseudo labels_. An interesing results was to find out that the model makes wrong guesses even in very confident predictions (>99%), possibly as a side effect of MixMatch's entropy minimization. Results are for only 200,000 update steps.

| | 4000 labels|
|:---|:---:|
|Plain MixMatch | 92.75 |
|Pseudo-labels (τ=0.95) | 92.25 |
|Pseudo-labels (τ=0.99) | 92.74
|Pseudo-labels (top 10% of each class) | 92.63 |


# References
```
@article{berthelot2019mixmatch,
  title={MixMatch: A Holistic Approach to Semi-Supervised Learning},
  author={Berthelot, David and Carlini, Nicholas and Goodfellow, Ian and Papernot, Nicolas and Oliver, Avital and Raffel, Colin},
  journal={arXiv preprint arXiv:1905.02249},
  year={2019}
}
```
