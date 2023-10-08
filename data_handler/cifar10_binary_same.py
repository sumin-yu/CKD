import os
import os.path
from PIL import Image
import numpy as np
import pickle
import torch

from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.utils import check_integrity, download_and_extract_archive
from data_handler.dataset_factory import GenericDataset
from data_handler.cifar10 import CIFAR_10S, CIFAR10

def rgb_to_grayscale(img):
    """Convert image to gray scale"""
    pil_gray_img = img.convert('L')
    np_gray_img = np.array(pil_gray_img, dtype=np.uint8)
    np_gray_img = np.dstack([np_gray_img, np_gray_img, np_gray_img])

    return np_gray_img

class CIFAR_10S_binary(CIFAR_10S):
    def __init__(self, root, split='train', transform=None,
                 seed=0, skewed_ratio=0.9):
        super(CIFAR_10S, self).__init__(root, transform=transform)

        self.split = split
        self.seed = seed
        self.test_pair = False

        self.num_classes = 2
        self.num_groups = 2

        imgs, inv_imgs, labels, colors, data_count = self._make_skewed(split, seed, skewed_ratio, self.num_classes)

        self.dataset = {}
        self.dataset['image'] = np.array(imgs)
        self.dataset['label'] = np.array(labels)
        self.dataset['color'] = np.array(colors)
        self.dataset['inv_image'] = np.array(inv_imgs)

        self._get_label_list()

        self.num_data = data_count
        
        self.features = [[int(s), int(l)] for s, l in zip(self.dataset['color'], self.dataset['label'])]
        self.n_data, self.idxs_per_group = self._data_count(self.features, self.num_groups, self.num_classes)

    def _make_skewed(self, split='train', seed=0, skewed_ratio=0.9, num_classes=2):

        train = False if split =='test' else True
        cifardata = CIFAR10('./data_cifar', train=train, shuffle=True, seed=seed, download=True, split=split)

        if split == 'train':
            num_data = 40000
        elif split == 'valid':
            # num_data = 20000
            num_data = 10000
        elif split == 'test':
            # num_data = 20000
            num_data = 10000
        # num_data = 50000 if split =='train' else 20000

        imgs = np.zeros((num_data, 32, 32, 3), dtype=np.uint8)
        inv_imgs = np.zeros((num_data, 32, 32, 3), dtype=np.uint8)
        labels = np.zeros(num_data)
        colors = np.zeros(num_data)
        data_count = np.zeros((2, num_classes), dtype=int)

        data_count_r = np.zeros((2, 10), dtype=int)
        # num_total_train_data = int((40000 // 10))
        # num_skewed_train_data = int((40000 * skewed_ratio) // 10)
        num_total_train_data = int((num_data // 10))
        num_skewed_train_data = int((num_data * skewed_ratio) // 10)

        for i, data in enumerate(cifardata):
            img, target = data

            if target < 5:
                if data_count_r[0, target] < (num_skewed_train_data):
                    imgs[i] = rgb_to_grayscale(img)
                    inv_imgs[i] = np.array(img)
                    colors[i] = 0
                    data_count[0, 0] += 1
                    data_count_r[0, target] += 1
                else:
                    imgs[i] = np.array(img)
                    inv_imgs[i] = rgb_to_grayscale(img)
                    colors[i] = 1
                    data_count[1, 0] += 1
                    data_count_r[1, target] += 1
                labels[i] = 0
            else:
                if data_count_r[0, target] < (num_total_train_data - num_skewed_train_data):
                    imgs[i] = rgb_to_grayscale(img)
                    inv_imgs[i] = np.array(img)
                    colors[i] = 0
                    data_count[0, 1] += 1
                    data_count_r[0, target] += 1
                else:
                    imgs[i] = np.array(img)
                    inv_imgs[i] = rgb_to_grayscale(img)
                    colors[i] = 1
                    data_count[1, 1] += 1
                    data_count_r[1, target] += 1
                labels[i] = 1

        print('<# of Skewed data>')
        print(data_count)
        print('<# of Skewed real data>')
        print(data_count_r)

        return imgs, inv_imgs,  labels, colors, data_count
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
        else: 
            # print(image)
            img_list = self.transform(image)
            input = img_list

        return input, 0, np.float32(color), np.int64(label), (index, 0)