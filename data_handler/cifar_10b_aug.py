import os
import os.path
from PIL import Image
import numpy as np
import pickle
import torch

from data_handler.cifar_10b import CIFAR_10B


class CIFAR_10B_aug(CIFAR_10B):
    def __init__(self, root, split='train', transform=None,
                 seed=0, skewed_ratio=0.8, editing_bias_alpha=0.0, test_set='original'):
        super(CIFAR_10B_aug, self).__init__(root, split=split, transform=transform, seed=seed, skewed_ratio=skewed_ratio, editing_bias_alpha=editing_bias_alpha, test_set=test_set)

    def __getitem__(self, index):
        image = self.dataset['image'][index]
        inv_image = self.dataset['inv_image'][index]
        label = self.dataset['label'][index]
        color = self.dataset['color'][index]
        org_noise = self.dataset['org_noise'][index]
        inv_noise = self.dataset['inv_noise'][index]

        if self.test_pair:
            img = self.transform(image)
            int_img = self.transform(inv_image)
            input = torch.stack([img, int_img])
            color = torch.Tensor([color, 0 if color==1 else 1])
            label = torch.Tensor([label, label])
        elif self.split == 'train' and self.test_mode == False:
            img = self.transform(image)
            img_edited = self.transform(inv_image)
            input = torch.stack([img, img_edited])
            color = torch.Tensor([color, 0 if color==1 else 1])
            label = torch.Tensor([label, label])
        else: 
            input = self.transform(image)

        return input, (org_noise, inv_noise), np.float32(color), np.int64(label), (index, 0)