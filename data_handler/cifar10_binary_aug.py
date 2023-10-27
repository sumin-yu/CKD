import os
import os.path
from PIL import Image
import numpy as np
import pickle
import torch

from data_handler.cifar10_binary import CIFAR_10S_binary


class CIFAR_10S_binary_aug(CIFAR_10S_binary):
    def __init__(self, root, split='train', transform=None,
                 seed=0, skewed_ratio=0.8, domain_gap_degree=0, 
                    editing_bias_alpha=0.0, editing_bias_beta=0, noise_degree=0,
                    noise_type='Spatter'):
        super(CIFAR_10S_binary_aug, self).__init__(root, split=split, transform=transform, seed=seed, skewed_ratio=skewed_ratio, domain_gap_degree=domain_gap_degree, 
                    editing_bias_alpha=editing_bias_alpha, editing_bias_beta=editing_bias_beta, noise_degree=noise_degree,
                    noise_type=noise_type)
    
    def __getitem__(self, index):
        image = self.dataset['image'][index]
        inv_image = self.dataset['inv_image'][index]
        label = self.dataset['label'][index]
        color = self.dataset['color'][index]

        if self.test_pair:
            img = self.transform(image)
            int_img = self.transform(inv_image)
            input = [img, int_img]
            input = torch.stack(input)
            color = torch.Tensor([color, 0 if color==1 else 1])
            label = torch.Tensor([label, label])
        elif self.split == 'train':
            img = self.transform(image)
            img_edited = self.transform(inv_image)
            input = [img, img_edited]
            input = torch.stack(input)
            color = torch.Tensor([color, 0 if color==1 else 1])
            label = torch.Tensor([label, label])
        else: 
            input = self.transform(image)

        return input, 0, np.float32(color), np.int64(label), (index, 0)