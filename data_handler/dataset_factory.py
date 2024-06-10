import torch.utils.data as data
import numpy as np
from collections import defaultdict
class DatasetFactory:
    def __init__(self):
        pass

    @staticmethod
    def get_dataset(name, transform=None, split='train', target='Blond_Hair', sensitive='Male', seed=0, skew_ratio=0.9, test_set='original', editing_bias_alpha=0.0, test_alpha_pc=False):

        if name == "celeba":
            from data_handler.celeba import CelebA
            root='./data/'
            return CelebA(root=root, split=split, transform=transform, target_attr=target, sen_attr=sensitive, test_set=test_set)
        elif name == "celeba_aug":
            from data_handler.celeba_aug import CelebA_aug
            root='./data/'
            return CelebA_aug(root=root, split=split, transform=transform, target_attr=target, sen_attr=sensitive, test_set=test_set)
        
        elif name == "lfw":
            from data_handler.lfw import LFWPeople
            root = './data/'
            return LFWPeople(root=root, split=split, image_set='funneled', transform=transform, target_attr=target, sen_attr=sensitive, test_set=test_set)
        elif name == "lfw_aug":
            from data_handler.lfw_aug import LFWPeople_aug
            root = './data/'
            return LFWPeople_aug(root=root, split=split, image_set='funneled', transform=transform, target_attr=target, sen_attr=sensitive, test_set=test_set)

        elif name == "cifar10_b_same":
            from data_handler.cifar10_binary_same import CIFAR_10S_binary_same
            root = './data_cifar'
            return CIFAR_10S_binary_same(root=root, split=split, transform=transform, seed=seed, skewed_ratio=skew_ratio, editing_bias_alpha=editing_bias_alpha, test_alpha_pc=test_alpha_pc)
        elif name == "cifar10_b_same_aug":
            from data_handler.cifar10_binary_same_aug import CIFAR_10S_binary_same_aug
            root = './data_cifar'
            return CIFAR_10S_binary_same_aug(root=root, split=split, transform=transform, seed=seed, skewed_ratio=skew_ratio, editing_bias_alpha=editing_bias_alpha, test_alpha_pc=test_alpha_pc)


class GenericDataset(data.Dataset):
    def __init__(self, root, split='train', transform=None, seed=0, uc=False):
        self.root = root
        self.split = split
        self.transform = transform
        self.seed = seed
        self.n_data = None
        self.uc = uc
        
    def __len__(self):
        return np.sum(self.n_data)
    
    def _data_count(self, features, n_groups, n_classes):
        idxs_per_group = defaultdict(lambda: [])
        data_count = np.zeros((n_groups, n_classes), dtype=int)
    
        if self.root == './data/jigsaw':
            for s, l in zip(self.g_array, self.y_array):
                data_count[s, l] += 1
        else:
            for idx, i in enumerate(features):
                s, l = int(i[0]), int(i[1])
                data_count[s, l] += 1
                idxs_per_group[(s,l)].append(idx)

            
        print(f'mode : {self.split}')        
        for i in range(n_groups):
            print('# of %d group data : '%i, data_count[i, :])
        return data_count, idxs_per_group
            
    def _make_data(self, features, n_groups, n_classes):
        # if the original dataset not is divided into train / test set, this function is used
        import copy
        min_cnt = 100
        data_count = np.zeros((n_groups, n_classes), dtype=int)
        tmp = []
        for i in reversed(self.features):
            s, l = int(i[0]), int(i[1])
            data_count[s, l] += 1
            if data_count[s, l] <= min_cnt:
                features.remove(i)
                tmp.append(i)
        
        train_data = features
        test_data = tmp
        return train_data, test_data
    

    def make_weights(self, method='kd_mfd', dataset='spucobirds', sampling='noBal'):
        if 'spucobirds' in dataset :
            weights = [1/self.group_weights[g,l] for g,l in zip(self.spurious, self.labels)]
        else:
            # if method == 'fairhsic' or 'kd_indiv_ukn3' or 'kd_mfd_ctf_ukn3':
            if sampling == 'cBal':
                group_weights = len(self) / self.n_data.sum(axis=0)
                weights = [group_weights[int(feature[1])] for feature in self.features]
            elif sampling == 'gcBal':
                group_weights = len(self) / self.n_data
                weights = [group_weights[int(feature[0]),int(feature[1])] for feature in self.features]
            if sampling == 'gBal':
                group_weights = len(self) / self.n_data.sum(axis=1)
                weights = [group_weights[int(feature[0])] for feature in self.features]
        return weights 
    
