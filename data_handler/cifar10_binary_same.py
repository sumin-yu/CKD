import os
import os.path
from PIL import Image
import numpy as np
import pickle
import torch

from data_handler.cifar10_binary import CIFAR10, CIFAR_10S_binary

class CIFAR_10S_binary_same(CIFAR_10S_binary):
    def __init__(self, root, split='train', transform=None,
                 seed=0, skewed_ratio=0.8, domain_gap_degree=0, editing_bias_alpha=0.0, editing_bias_beta=0, noise_degree=0, noise_type='Spatter'):
        super(CIFAR_10S_binary_same, self).__init__(root, split=split, transform=transform, seed=seed, skewed_ratio=skewed_ratio, domain_gap_degree=domain_gap_degree, editing_bias_alpha=editing_bias_alpha, editing_bias_beta=editing_bias_beta, noise_degree=noise_degree, noise_type=noise_type)

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
            imgs[i], inv_imgs[i], org_noise, inv_noise = self._make_editing_bias(imgs[i], inv_imgs[i], colors[i], data_count_r[int(colors[i]), target_r], data_count_r_answer[int(colors[i]), target_r])

            ### for test ###
            if org_noise == 1:
                data_count_r_org[int(colors[i]), target_r] += 1
            if inv_noise == 1:
                data_count_r_inv[int(colors[i]), target_r] += 1

        data_count[:,0] = np.sum(data_count_r[:,:5], axis=1)
        data_count[:,1] = np.sum(data_count_r[:,5:], axis=1)
        print('<# of Skewed data>')
        print(data_count)
        print('<# of Skewed real data>')
        print(data_count_r)

        ### for test ###
        if self.editing_bias_alpha != 0:
            # save data_count_r_org as txt file in one file with the name of 'org'
            if not os.path.exists('./data_cifar/editing_bias_same_{}_{}.txt'.format(self.editing_bias_alpha, self.editing_bias_beta)):
                # open file
                f = open('./data_cifar/editing_bias_same_{}_{}.txt'.format(self.editing_bias_alpha, self.editing_bias_beta), 'w')
                # write data
                f.write('org data num\n')
                f.write(str(data_count_r))
                f.write('\ngray wo noise & color w noise\n')
                tmp = np.vstack((data_count_r[0]-data_count_r_org[0], data_count_r_org[1]))
                f.write(str(tmp))
                f.write('\ngray_inv w noise & color_inv wo noise\n')
                tmp = np.vstack((data_count_r_inv[0], data_count_r[1]-data_count_r_inv[1]))
                f.write(str(tmp))
                # close file
                f.close()           

            print('<# of org noised data>')
            print(data_count_r_org)
            print('<# of inv noised data>')
            print(data_count_r_inv)

        return imgs, inv_imgs,  labels, colors, data_count