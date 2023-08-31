import torch.utils.data as data
import numpy as np
from collections import defaultdict
class DatasetFactory:
    def __init__(self):
        pass

    @staticmethod
    def get_dataset(name, transform=None, split='train', target='Blond_Hair', sensitive='Male', seed=0, skew_ratio=1., labelwise=False, method=None,num_aug=1):

        if name == "utkface":
            from data_handler.utkface import UTKFaceDataset
            root = './data/UTKFace'
            return UTKFaceDataset(root=root, split=split, transform=transform,
                                  labelwise=labelwise)

        elif name == "celeba":
            from data_handler.celeba import CelebA
            root='./data/'
            return CelebA(root=root, split=split, transform=transform, target_attr=target, sen_attr=sensitive)
        elif name == "celeba_aug":
            from data_handler.celeba_aug import CelebA_aug
            root='./data/'
            return CelebA_aug(root=root, split=split, transform=transform, target_attr=target, sen_attr=sensitive,num_aug=num_aug)
        elif name == "celeba_aug2":
            from data_handler.celeba_aug2 import CelebA_aug
            root='./data/'
            return CelebA_aug(root=root, split=split, transform=transform, target_attr=target, sen_attr=sensitive)
        elif name == "celeba_aug3":
            from data_handler.celeba_aug3 import CelebA_aug
            root='./data/'
            return CelebA_aug(root=root, split=split, transform=transform, target_attr=target, sen_attr=sensitive)
        elif name == 'celeba_aug_ukn':
            from data_handler.celeba_aug_ukn import CelebA_aug
            root='./data/'
            return CelebA_aug(root=root, split=split, transform=transform, target_attr=target, sen_attr=sensitive, method=method)
        elif name == 'celeba_aug_ukn_wo_org':
            from data_handler.celeba_aug_ukn_wo_org import CelebA_aug
            root='./data/'
            return CelebA_aug(root=root, split=split, transform=transform, target_attr=target, sen_attr=sensitive)

        
        elif name == "cifar10":
            from data_handler.cifar10 import CIFAR_10S
            root = './data_cifar'
            return CIFAR_10S(root=root, split=split, transform=transform, seed=seed, skewed_ratio=skew_ratio)
        elif name == "cifar10_aug":
            from data_handler.cifar10_aug import CIFAR_10S
            root = './data_cifar'
            return CIFAR_10S(root=root, split=split, transform=transform, seed=seed, skewed_ratio=skew_ratio)
        
        elif name == "cifar10_all":
            from data_handler.cifar10_all import CIFAR_10S
            root = './data_cifar'
            return CIFAR_10S(root=root, split=split, transform=transform, seed=seed, skewed_ratio=skew_ratio,
                                labelwise=labelwise)
        
        elif name == "spucobirds":
            from data_handler.spucobirds import SpuCoBirds
            root = './data/spuco'
            dataset =  SpuCoBirds(root=root, split=split, transform=transform)
            dataset.initialize()
            return dataset
        elif name == 'spucobirds_aug':
            from data_handler.spucobirds_aug import SpuCoBirds_aug
            root = './data/spuco'
            dataset = SpuCoBirds_aug(root=root, split=split, transform=transform,num_aug=num_aug)
            dataset.initialize()
            return dataset



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
    

    def make_weights(self, method='kd_mfd', dataset='spucobirds'):
        if 'spucobirds' in dataset :
            weights = [1/self.group_weights[g,l] for g,l in zip(self.spurious, self.labels)]
        elif self.root != './data/jigsaw':
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
    
