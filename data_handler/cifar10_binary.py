import os
import os.path
from PIL import Image
import numpy as np
import pickle
import torch
from skimage.filters import gaussian
import cv2

from data_handler.corrupted_cifar10_protocol import CORRUPTED_CIFAR10_PROTOCOL
from data_handler.cifar10 import CIFAR10, CIFAR_10S

class CIFAR_10S_binary(CIFAR_10S):
    def __init__(self, root, split='train', transform=None,
                 seed=0, skewed_ratio=0.8,
                    domain_gap_degree=0,
                    editing_bias_alpha=0.0, editing_bias_beta=0, noise_degree=0,
                    noise_type='Spatter', group_bias_type='color', group_bias_degree=1, noise_corr='neg', test_alpha_pc=False, test_beta2_pc=False):
        super(CIFAR_10S, self).__init__(root, split=split, transform=transform, seed=seed)

        self.test_pair = False
        self.test_mode = False # mode change for training set (just get org image)
        self.test_alpha_pc = test_alpha_pc
        self.test_beta2_pc = test_beta2_pc
        self.split = split
        self.seed = seed

        self.num_classes = 2
        self.num_groups = 2

        self.noise_injection = CORRUPTED_CIFAR10_PROTOCOL[noise_type]
        self.domain_gap_degree = domain_gap_degree
        self.editing_bias_alpha = editing_bias_alpha
        self.noise_degree = noise_degree
        self.noise_corr = noise_corr

        self.group_bias_type=group_bias_type
        self.group_bias_degree=group_bias_degree

        if self.editing_bias_alpha != 0:
            if editing_bias_beta == 0:
                self.editing_bias_beta = 0.0
            elif editing_bias_beta == 1:
                self.editing_bias_beta = self.editing_bias_alpha * 0.5
            elif editing_bias_beta == 2:
                self.editing_bias_beta = self.editing_bias_alpha
            elif editing_bias_beta == 3:
                self.editing_bias_beta = 1.0
            else:
                self.editing_bias_beta = editing_bias_beta # beta =4 for test 

        imgs, intervened_imgs, labels, colors, data_count, org_noise, inv_noise = self._make_skewed(split, seed, skewed_ratio)

        self.dataset = {}
        self.dataset['image'] = np.array(imgs)
        self.dataset['label'] = np.array(labels)
        self.dataset['color'] = np.array(colors)
        self.dataset['inv_image'] = np.array(intervened_imgs)
        self.dataset['org_noise'] = np.array(org_noise)
        self.dataset['inv_noise'] = np.array(inv_noise)

        ### save 10 images as 1 image for test if there is no saved image file ###
        if self.domain_gap_degree != 0:
            if not os.path.exists('./data_cifar/domain_gap_{}_{}.png'.format(noise_type, domain_gap_degree)):
                for i in range(10): 
                    if i == 0:
                        img = imgs[i]
                        inv_img = intervened_imgs[i]
                    else:
                        img = np.concatenate((img, imgs[i]), axis=1)
                        inv_img = np.concatenate((inv_img, intervened_imgs[i]), axis=1)
                img = np.concatenate((img, inv_img), axis=0)
                img = Image.fromarray(img)
                img.save('./data_cifar/domain_gap_{}_{}.png'.format(noise_type, domain_gap_degree))

        self._get_label_list()

        self.num_data = data_count
        
        self.features = [[int(s), int(l), 0] for s, l in zip(self.dataset['color'], self.dataset['label'])]
        self.n_data, self.idxs_per_group = self._data_count(self.features, self.num_groups, self.num_classes)

    def _make_editing_bias(self, img, inv_img, color, num, tot_num):
        org_noise = 0
        inv_noise = 0
        if self.split == 'test' and self.test_beta2_pc:
            self.editing_bias_beta = self.editing_bias_alpha

        if self.editing_bias_alpha != 0:
            if (self.split != 'test' and self.editing_bias_beta != 0) or (self.split == 'test' and self.test_beta2_pc):
                if color == 0: # org color is gray
                    if num > tot_num*self.editing_bias_alpha*(1-self.editing_bias_beta) and num <= tot_num*(1-(1-self.editing_bias_alpha)*(1-self.editing_bias_beta)):
                        inv_img = self.noise_injection(inv_img, severity=self.noise_degree, seed=self.seed)
                        inv_noise = 1
                    if num > tot_num*self.editing_bias_alpha:
                        img = self.noise_injection(img, severity=self.noise_degree, seed=self.seed)
                        org_noise = 1
                else: # org color is color
                    if num <= tot_num*self.editing_bias_alpha:
                        img = self.noise_injection(img, severity=self.noise_degree, seed=self.seed)
                        org_noise = 1
                    if num <= tot_num*self.editing_bias_alpha*(1-self.editing_bias_beta) or num > tot_num*(1-(1-self.editing_bias_alpha)*(1-self.editing_bias_beta)):
                        inv_img = self.noise_injection(inv_img, severity=self.noise_degree, seed=self.seed)
                        inv_noise = 1
            else:
                if self.test_alpha_pc:
                    if color == 0:
                        if num > tot_num*self.editing_bias_alpha:
                            img = self.noise_injection(img, severity=self.noise_degree, seed=self.seed)
                            org_noise = 1
                        else:
                            inv_img = self.noise_injection(inv_img, severity=self.noise_degree, seed=self.seed)
                            inv_noise = 1
                    else:
                        if num <= tot_num*self.editing_bias_alpha:
                            img = self.noise_injection(img, severity=self.noise_degree, seed=self.seed)
                            org_noise = 1
                        else:
                            inv_img = self.noise_injection(inv_img, severity=self.noise_degree, seed=self.seed)
                            inv_noise = 1
                else:
                    if color == 0: # org color is gray
                        if num > tot_num*self.editing_bias_alpha:
                            img = self.noise_injection(img, severity=self.noise_degree, seed=self.seed)
                            inv_img = self.noise_injection(inv_img, severity=self.noise_degree, seed=self.seed)
                            org_noise = 1
                            inv_noise = 1
                    else: # org color is color
                        if num <= tot_num*self.editing_bias_alpha:
                            img = self.noise_injection(img, severity=self.noise_degree, seed=self.seed)
                            inv_img = self.noise_injection(inv_img, severity=self.noise_degree, seed=self.seed)
                            org_noise = 1
                            inv_noise = 1
        return img, inv_img, org_noise, inv_noise

    def _make_domain_gap(self, img):
        if self.domain_gap_degree != 0:
            domain_gap_img = self.noise_injection(img, severity=self.domain_gap_degree, seed=self.seed)
            domain_gap_img =  domain_gap_img.astype(np.uint8)
        else:
            domain_gap_img = img
        return domain_gap_img

    def _set_data(self, img, group):
        if self.noise_corr == 'neg':
            if self.split != 'test':
                if group==1:    
                    return np.array(img), self._make_domain_gap(self._perturbing(img, bias_type=self.group_bias_type, degree=self.group_bias_degree)), group
                elif group==0:
                    return self._perturbing(img, bias_type=self.group_bias_type, degree=self.group_bias_degree), self._make_domain_gap(np.array(img)), group
            elif self.test_alpha_pc == False:
                if group==1:    
                    return np.array(img), self._perturbing(img, bias_type=self.group_bias_type, degree=self.group_bias_degree), group
                elif group==0:
                    return self._perturbing(img, bias_type=self.group_bias_type, degree=self.group_bias_degree), np.array(img), group
            else:
                if group==1:
                    return np.array(img), np.array(img), group
                elif group==0:
                    return self._perturbing(img, bias_type=self.group_bias_type, degree=self.group_bias_degree), self._perturbing(img, bias_type=self.group_bias_type, degree=self.group_bias_degree), group
        elif self.noise_corr == 'pos':
            if self.split != 'test':
                if group==1:    
                    return self._perturbing(img, bias_type=self.group_bias_type, degree=self.group_bias_degree), self._make_domain_gap(np.array(img)), group
                elif group==0:
                    return np.array(img), self._make_domain_gap(self._perturbing(img, bias_type=self.group_bias_type, degree=self.group_bias_degree)), group
            elif self.test_alpha_pc == False:   
                if group==1:    
                    return self._perturbing(img, bias_type=self.group_bias_type, degree=self.group_bias_degree), np.array(img), group
                elif group==0:
                    return np.array(img), self._perturbing(img, bias_type=self.group_bias_type, degree=self.group_bias_degree), group
            else:
                if group==1:
                    return self._perturbing(img, bias_type=self.group_bias_type, degree=self.group_bias_degree), self._perturbing(img, bias_type=self.group_bias_type, degree=self.group_bias_degree), group
                elif group==0:
                    return np.array(img), np.array(img), group

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
        labels_r = np.zeros(num_data)
        colors = np.zeros(num_data)
        org_noise = np.zeros(num_data)
        inv_noise = np.zeros(num_data)
        data_count = np.zeros((2, num_classes), dtype=int)

        data_count_r = np.zeros((2, 10), dtype=int)
        data_count_r_answer = np.zeros((2, 10), dtype=int)
        num_total_train_data = int((num_data // 10))
        num_skewed_train_data = int((num_data * skewed_ratio) // 10)
        if skewed_ratio == 0.8:
            off_set = int((num_data * 0.2) // 10)
        elif skewed_ratio == 0.9:
            off_set = int((num_data * 0.1) // 10)
        else:
            raise ValueError('skewed_ratio should be 0.8 or 0.9')

        ### for test ###
        data_count_r_org = np.zeros((2, 10), dtype=int)
        data_count_r_inv = np.zeros((2, 10), dtype=int)

        data_count_r_answer[0, :2] = num_skewed_train_data + off_set
        data_count_r_answer[1, :2] = num_total_train_data - (num_skewed_train_data + off_set)
        data_count_r_answer[0, 2] = num_skewed_train_data
        data_count_r_answer[1, 2] = num_total_train_data - num_skewed_train_data
        data_count_r_answer[0, 3:5] = num_skewed_train_data - off_set
        data_count_r_answer[1, 3:5] = num_total_train_data - (num_skewed_train_data - off_set)
        data_count_r_answer[0, 5:7] = num_total_train_data - (num_skewed_train_data + off_set)
        data_count_r_answer[1, 5:7] = num_skewed_train_data + off_set
        data_count_r_answer[0, 7] = num_total_train_data - num_skewed_train_data
        data_count_r_answer[1, 7] = num_skewed_train_data
        data_count_r_answer[0, 8:] = num_total_train_data - (num_skewed_train_data - off_set)
        data_count_r_answer[1, 8:] = num_skewed_train_data - off_set

        for i, data in enumerate(cifardata):
            img, target_r = data
            labels_r[i] = target_r
            if target_r < 5:
                labels[i] = 0
                if target_r < 2:
                    if data_count_r[0, target_r] < (num_skewed_train_data + off_set):
                        imgs[i], inv_imgs[i], colors[i] = self._set_data(img, group=0)
                    else:
                        imgs[i], inv_imgs[i], colors[i] = self._set_data(img, group=1)
                elif target_r == 2:
                    if data_count_r[0, target_r] < (num_skewed_train_data):
                        imgs[i], inv_imgs[i], colors[i] = self._set_data(img, group=0)
                    else:
                        imgs[i], inv_imgs[i], colors[i] = self._set_data(img, group=1)
                else:
                    if data_count_r[0, target_r] < (num_skewed_train_data - off_set):
                        imgs[i], inv_imgs[i], colors[i] = self._set_data(img, group=0)
                    else:
                        imgs[i], inv_imgs[i], colors[i] = self._set_data(img, group=1)
            else:
                labels[i] = 1
                if target_r < 7:
                    if data_count_r[0, target_r] < (num_total_train_data - num_skewed_train_data - off_set):
                        imgs[i], inv_imgs[i], colors[i] = self._set_data(img, group=0)
                    else:
                        imgs[i], inv_imgs[i], colors[i] = self._set_data(img, group=1)
                elif target_r == 7:
                    if data_count_r[0, target_r] < (num_total_train_data - num_skewed_train_data):
                        imgs[i], inv_imgs[i], colors[i] = self._set_data(img, group=0)
                    else:
                        imgs[i], inv_imgs[i], colors[i] = self._set_data(img, group=1)
                else:
                    if data_count_r[0, target_r] < (num_total_train_data - num_skewed_train_data + off_set):
                        imgs[i], inv_imgs[i], colors[i] = self._set_data(img, group=0)
                    else:
                        imgs[i], inv_imgs[i], colors[i] = self._set_data(img, group=1)
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

        return imgs, inv_imgs, labels, colors, data_count, org_noise, inv_noise

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
            color = torch.Tensor([color, 0 if color ==1 else 1])
            label = torch.Tensor([label, label])
        else: 
            img_list = self.transform(image)
            input = img_list

        return input,  (org_noise, inv_noise), np.float32(color), np.int64(label), (index, 0)