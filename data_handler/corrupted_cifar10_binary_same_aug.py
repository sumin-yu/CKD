import os
import os.path
from PIL import Image
import numpy as np
import pickle
import torch

from data_handler.cifar10_binary_same_aug import CIFAR_10S_binary_same_aug

class Corrupted_CIFAR_10S_binary_same_aug(CIFAR_10S_binary_same_aug):
    def __init__(self, root, split='train', transform=None,
                 seed=0, skewed_ratio=0.8):
        super(CIFAR_10S_binary_same_aug, self).__init__(root, split=split, transform=transform, seed=seed, skewed_ratio=skewed_ratio)

    def _perturbing(self, x, severity=1):
        x =np.array(x, dtype=np.uint8)
        c = [0.04, 0.06, 0.08, 0.09, 0.10][severity - 1]
        x = np.array(x) / 255.0
        return np.clip(x + np.random.normal(size=x.shape, scale=c), 0, 1) * 255
