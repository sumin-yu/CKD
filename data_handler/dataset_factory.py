import torch.utils.data as data
import numpy as np
from collections import defaultdict
class DatasetFactory:
    def __init__(self):
        pass

    @staticmethod
    def get_dataset(name, transform=None, split='train', target='Attractive', seed=0, skew_ratio=1., labelwise=False, num_aug=1, tuning=False):

        if name == "utkface":
            from data_handler.utkface import UTKFaceDataset
            root = './data/UTKFace'
            return UTKFaceDataset(root=root, split=split, transform=transform,
                                  labelwise=labelwise)

        elif name == "celeba":
            from data_handler.celeba import CelebA
            root='./data/'
            return CelebA(root=root, split=split, transform=transform, target_attr=target)
        
        elif name == "cifar10":
            from data_handler.cifar10 import CIFAR_10S
            root = './data/cifar10'
            return CIFAR_10S(root=root, split=split, transform=transform, seed=seed, skewed_ratio=skew_ratio,
                              tuning=tuning)
        elif name == "cifar10_indiv":
            from data_handler.cifar10_indiv import CIFAR_10S
            root = './data/cafar10'
            return CIFAR_10S(root=root, split=split, transform=transform, seed=seed, skewed_ratio=skew_ratio,
                             num_aug=num_aug, tuning=tuning)


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
    

    def make_weights(self, method):
        if self.root != './data/jigsaw':
            if method == 'fairhsic':
                group_weights = len(self) / self.n_data.sum(axis=0)
                weights = [group_weights[int(feature[1])] for feature in self.features]
#             elif method == 'cgdro_new':
#                 weights = self.n_data.sum(axis=0) / self.n_data
#                 weights = [group_weights[int(feature[0]),int(feature[1])] for feature in self.features] 
            else:
                group_weights = len(self) / self.n_data
                weights = [group_weights[int(feature[0]),int(feature[1])] for feature in self.features]
        else:
            if method == 'fairhsic':
                group_weights = len(self) / self.n_data.sum(axis=0)
                weights = [group_weights[l] for g,l in zip(self.g_array,self.y_array)]
#             elif method == 'cgdro_new':
#                 weights = self.n_data.sum(axis=0) / self.n_data
#                 weights = [group_weights[g,l] for g,l in zip(self.g_array,self.y_array)]
            else:
                group_weights = len(self) / self.n_data
                weights = [group_weights[g,l] for g,l in zip(self.g_array,self.y_array)]
        return weights 
    