import os
import os.path
from PIL import Image
import numpy as np
import pickle
import torch

from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.utils import check_integrity, download_and_extract_archive
from data_handler.dataset_factory import GenericDataset
from data_handler.cifar10 import CIFAR_10S


def rgb_to_grayscale(img):
    """Convert image to gray scale"""
    pil_gray_img = img.convert('L')
    np_gray_img = np.array(pil_gray_img, dtype=np.uint8)
    np_gray_img = np.dstack([np_gray_img, np_gray_img, np_gray_img])

    return np_gray_img


class CIFAR_10S_aug(CIFAR_10S):
    def __init__(self, root, split='train', transform=None,
                 seed=0, skewed_ratio=0.8):
        super(CIFAR_10S_aug, self).__init__(root=root, split=split, transform=transform, seed=seed, skewed_ratio=skewed_ratio)

        self.num_aug = 1

    def __getitem__(self, index):
        image = self.dataset['image'][index]
        label = self.dataset['label'][index]
        color = self.dataset['color'][index]
        intervened_image = self.dataset['intervened_image'][index]  

        if self.split == 'train' :
            img_list = []
            intervened_img_list = []
            for _ in range(self.num_aug):
                img_list.append(self.transform(image))
                intervened_img_list.append(self.transform(intervened_image))
            # if self.target_transform:
            #     label = self.target_transform(label)
            input = torch.stack(img_list + intervened_img_list)
            # print(input.size())

        elif self.test_pair:
            img = self.transform(image)
            int_img = self.dataset['image'][int(index+1-2*(index%2))]
            int_img = self.transform(int_img)
            input = [img, int_img]
            input = torch.stack(input)

        else: # valid or test
            img_list = self.transform(image)
            input = img_list

        return input, 0, np.float32(color), np.int64(label), index
        
        # return image, intervened_image, 0, np.float32(color), np.int64(label), (index, 0)