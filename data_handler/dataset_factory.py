import torch.utils.data as data
import numpy as np
from collections import defaultdict
class DatasetFactory:
    def __init__(self):
        pass

    @staticmethod
    def get_dataset(name, transform=None, split='train', target='Blond_Hair', sensitive='Male', seed=0, skew_ratio=0.9, sampling='noBal', method=None,num_aug=1, test_set='original', domain_gap_degree=0, editing_bias_alpha=0.0, editing_bias_beta=0, noise_degree=0, noise_type='Spatter', group_bias_type='color', group_bias_degree=1, noise_corr='neg', test_alpha_pc=False, test_beta2_pc=False):

        if name == "utkface":
            from data_handler.utkface import UTKFaceDataset
            root = './data/UTKFace'
            return UTKFaceDataset(root=root, split=split, transform=transform,
                                  sampling=sampling)

        elif name == "celeba":
            from data_handler.celeba import CelebA
            root='./data/'
            return CelebA(root=root, split=split, transform=transform, target_attr=target, sen_attr=sensitive, test_set=test_set)
        elif name == "celeba_pseudo":
            from data_handler.celeba_pseudo import CelebA_pseudo
            root='./data/'
            return CelebA_pseudo(root=root, split=split, transform=transform, target_attr=target, sen_attr=sensitive)
        elif name == "celeba_aug":
            from data_handler.celeba_aug import CelebA_aug
            root='./data/'
            return CelebA_aug(root=root, split=split, transform=transform, target_attr=target, sen_attr=sensitive,num_aug=num_aug, test_set=test_set)
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
        elif name == 'celeba_aug_filtered':
            from data_handler.celeba_aug_filtered import CelebA_aug
            root='./data/'
            return CelebA_aug(root=root, split=split, transform=transform, target_attr=target, sen_attr=sensitive)


        
        elif name == "cifar10":
            from data_handler.cifar10 import CIFAR_10S
            root = './data_cifar'
            return CIFAR_10S(root=root, split=split, transform=transform, seed=seed, skewed_ratio=skew_ratio)
        elif name == "cifar10_aug":
            from data_handler.cifar10_aug import CIFAR_10S_aug
            root = './data_cifar'
            return CIFAR_10S_aug(root=root, split=split, transform=transform, seed=seed, skewed_ratio=skew_ratio)
        
        elif name == "cifar10_b":
            from data_handler.cifar10_binary import CIFAR_10S_binary
            root = './data_cifar'
            return CIFAR_10S_binary(root=root, split=split, transform=transform, seed=seed, skewed_ratio=skew_ratio, domain_gap_degree=domain_gap_degree, editing_bias_alpha=editing_bias_alpha, editing_bias_beta=editing_bias_beta, noise_degree=noise_degree, noise_type=noise_type, group_bias_type=group_bias_type, group_bias_degree=group_bias_degree, noise_corr=noise_corr, test_alpha_pc=test_alpha_pc, test_beta2_pc=test_beta2_pc)
        elif name == "cifar10_b_aug":
            from data_handler.cifar10_binary_aug import CIFAR_10S_binary_aug
            root = './data_cifar'
            return CIFAR_10S_binary_aug(root=root, split=split, transform=transform, seed=seed, skewed_ratio=skew_ratio, domain_gap_degree=domain_gap_degree, editing_bias_alpha=editing_bias_alpha, editing_bias_beta=editing_bias_beta, noise_degree=noise_degree, noise_type=noise_type, group_bias_type=group_bias_type, group_bias_degree=group_bias_degree, noise_corr=noise_corr, test_alpha_pc=test_alpha_pc, test_beta2_pc=test_beta2_pc)
        elif name == "cifar10_b_same":
            from data_handler.cifar10_binary_same import CIFAR_10S_binary_same
            root = './data_cifar'
            return CIFAR_10S_binary_same(root=root, split=split, transform=transform, seed=seed, skewed_ratio=skew_ratio, domain_gap_degree=domain_gap_degree, editing_bias_alpha=editing_bias_alpha, editing_bias_beta=editing_bias_beta, noise_degree=noise_degree, noise_type=noise_type, group_bias_type=group_bias_type, group_bias_degree=group_bias_degree, noise_corr=noise_corr,   test_alpha_pc=test_alpha_pc, test_beta2_pc=test_beta2_pc)
        elif name == "cifar10_b_same_aug":
            from data_handler.cifar10_binary_same_aug import CIFAR_10S_binary_same_aug
            root = './data_cifar'
            return CIFAR_10S_binary_same_aug(root=root, split=split, transform=transform, seed=seed, skewed_ratio=skew_ratio, domain_gap_degree=domain_gap_degree, editing_bias_alpha=editing_bias_alpha, editing_bias_beta=editing_bias_beta, noise_degree=noise_degree, noise_type=noise_type, group_bias_type=group_bias_type, group_bias_degree=group_bias_degree, noise_corr=noise_corr, test_alpha_pc=test_alpha_pc, test_beta2_pc=test_beta2_pc)

        elif name == "Ccifar10_b":
            from data_handler.corrupted_cifar10_binary import Corrupted_CIFAR_10S_binary
            root = './data_cifar'
            return Corrupted_CIFAR_10S_binary(root=root, split=split, transform=transform, seed=seed, skewed_ratio=skew_ratio)
        elif name == "Ccifar10_b_aug":
            from data_handler.corrupted_cifar10_binary_aug import Corrupted_CIFAR_10S_binary_aug
            root = './data_cifar'
            return Corrupted_CIFAR_10S_binary_aug(root=root, split=split, transform=transform, seed=seed, skewed_ratio=skew_ratio)
        elif name == "Ccifar10_b_same":
            from data_handler.corrupted_cifar10_binary_same import Corrupted_CIFAR_10S_binary_same
            root = './data_cifar'
            return Corrupted_CIFAR_10S_binary_same(root=root, split=split, transform=transform, seed=seed, skewed_ratio=skew_ratio)
        elif name == "Ccifar10_b_same_aug":
            from data_handler.corrupted_cifar10_binary_same_aug import Corrupted_CIFAR_10S_binary_same_aug
            root = './data_cifar'
            return Corrupted_CIFAR_10S_binary_same_aug(root=root, split=split, transform=transform, seed=seed, skewed_ratio=skew_ratio)


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
        elif name == "spucobirds_aug_filtered":
            from data_handler.spucobirds_aug_filtered import SpuCoBirds_aug
            root = './data/spuco'
            dataset = SpuCoBirds_aug(root=root, split=split, transform=transform)
            dataset.initialize()
            return dataset
        
        elif name =='raf':
            from data_handler.raf import RAF
            root = './data/rafdb'
            return RAF(root=root, split=split, transform=transform)
        elif name =='raf_aug':
            from data_handler.raf_aug import RAF_aug
            root = './data/rafdb'
            return RAF_aug(root=root, split=split, transform=transform, num_aug=num_aug)


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
    
