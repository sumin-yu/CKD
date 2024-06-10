import os
import os.path
from PIL import Image
import numpy as np
import pickle
import torch

from data_handler.cifar10_binary import CIFAR10, CIFAR_10S_binary

class CIFAR_10S_binary_same(CIFAR_10S_binary):
    def __init__(self, root, split='train', transform=None,
                 seed=0, skewed_ratio=0.8, editing_bias_alpha=0.0, test_alpha_pc=False):
        super(CIFAR_10S_binary_same, self).__init__(root, split=split, transform=transform, seed=seed, skewed_ratio=skewed_ratio, editing_bias_alpha=editing_bias_alpha, test_alpha_pc=test_alpha_pc)

    def _make_skewed(self, split='train', seed=0, skewed_ratio=0.8, num_classes=2):

        train = False if split =='test' else True
        cifardata = CIFAR10('./data_cifar', train=train, shuffle=True, seed=seed, download=True, split=split)

        if split == 'train':
            num_data = 40000
        elif split == 'valid':
            num_data = 10000
        elif split == 'test':
            num_data = 10000

        imgs = np.zeros((num_data, 32, 32, 3), dtype=np.uint8)
        inv_imgs = np.zeros((num_data, 32, 32, 3), dtype=np.uint8)
        labels = np.zeros(num_data)
        colors = np.zeros(num_data)
        org_noise = np.zeros(num_data)
        inv_noise = np.zeros(num_data)
        data_count = np.zeros((2, num_classes), dtype=int)

        data_count_r = np.zeros((2, 10), dtype=int)
        data_count_r_answer = np.zeros((2, 10), dtype=int)
        num_total_train_data = int((num_data // 10))
        num_skewed_train_data = int((num_data * skewed_ratio) // 10)

        ### for test ###
        data_count_r_org = np.zeros((2, 10), dtype=int)
        data_count_r_inv = np.zeros((2, 10), dtype=int)

        data_count_r_answer[0, :5] = num_skewed_train_data
        data_count_r_answer[1, :5] = num_total_train_data - num_skewed_train_data
        data_count_r_answer[0, 5:] = num_total_train_data - num_skewed_train_data
        data_count_r_answer[1, 5:] = num_skewed_train_data

        for i, data in enumerate(cifardata):
            img, target_r = data

            if target_r < 5:
                if data_count_r[0, target_r] < (num_skewed_train_data):
                    imgs[i], inv_imgs[i], colors[i] = self._set_data(img, group=0)
                else:
                    imgs[i], inv_imgs[i], colors[i] = self._set_data(img, group=1)
                labels[i] = 0
            else:
                if data_count_r[0, target_r] < (num_total_train_data - num_skewed_train_data):
                    imgs[i], inv_imgs[i], colors[i] = self._set_data(img, group=0)
                else:
                    imgs[i], inv_imgs[i], colors[i] = self._set_data(img, group=1)
                labels[i] = 1
            data_count_r[int(colors[i]), target_r] += 1
            imgs[i], inv_imgs[i], org_noise_, inv_noise_ = self._make_editing_bias(imgs[i], inv_imgs[i], colors[i], data_count_r[int(colors[i]), target_r], data_count_r_answer[int(colors[i]), target_r])
            org_noise[i] = org_noise_
            inv_noise[i] = inv_noise_

            ### for test ###
            data_count_r_org[int(colors[i]), target_r] += org_noise_
            data_count_r_inv[int(colors[i]), target_r] += inv_noise_

        data_count[:,0] = np.sum(data_count_r[:,:5], axis=1)
        data_count[:,1] = np.sum(data_count_r[:,5:], axis=1)
        print('----------------------', split, ' data distribution discription----------------------')
        print('<# of Skewed data>')
        print(data_count)
        print('<# of Skewed real data>')
        print(data_count_r)
        print('answer')
        print(data_count_r_answer)

        ### for test ###
        if self.editing_bias_alpha != 0:
            print('---------------------------------')
            print('<# of org no-noised data of group0')
            print(data_count_r_answer[0] - data_count_r_org[0])
            print('<# of inv noised data of group0')
            print(data_count_r_inv[0])
            print('---------------------------------')
            print('<# of org noised data of group1')
            print(data_count_r_org[1])
            print('<# of inv no-noised data of group1')
            print(data_count_r_answer[1] - data_count_r_inv[1])

        return imgs, inv_imgs,  labels, colors, data_count, org_noise, inv_noise