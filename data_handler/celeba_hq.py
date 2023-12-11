import torch
import os
from os.path import join
import PIL
from PIL import ImageOps, Image
import pandas
import numpy as np
import zipfile
from functools import partial
from torchvision.datasets.utils import verify_str_arg
from data_handler.dataset_factory import GenericDataset

class CelebA_HQ(GenericDataset):

    def __init__(self, root, split="train", target_type="attr", transform=None,
                 target_transform=None, download=False, target_attr='Blond_Hair', sen_attr='Male',num_aug=1, test_set='original', test_pc_G=None):
        super(CelebA_HQ, self).__init__(root, transform=transform)
        self.split = split
        self.num_aug=num_aug
        self.test_pair = False
        self.ctf_dir = "img_align_celeba_edited_Male".format(sen_attr) # should be change to correctly evaluate pc
        self.test_pc_G = test_pc_G
        if test_pc_G is not None:
            if test_pc_G == 'Hair_Length':
                self.G1_dir = "img_align_celeba_edited_{}".format('Long_Hair')
                self.G2_dir = "img_align_celeba_edited_{}".format('Short_Hair')
            elif test_pc_G == 'Hair_Curl':
                self.G1_dir = "img_align_celeba_edited_{}".format('Curly_Hair')
                self.G2_dir = "img_align_celeba_edited_{}".format('Straight_Hair')

        if isinstance(target_type, list):
            self.target_type = target_type
        else:
            self.target_type = [target_type]

        if not self.target_type and self.target_transform is not None:
            raise RuntimeError('target_transform is specified but target_type is empty')

        # SELECT the features
        self.sensitive_attr = sen_attr
        self.target_attr = target_attr
        split_map = {
            "train": 0,
            "valid": 1,
            "test": 2,
            "all": None,
        }
        split = split_map[verify_str_arg(split.lower(), "split",
                                         ("train", "valid", "test", "all" ))]
        
        fn = partial(join, self.root)
        if test_set == 'pc_G':
            hq = pandas.read_csv(fn("image_list_test_pc_G_filter_{}.txt".format(test_pc_G)), delim_whitespace=True)
        else:
            hq = pandas.read_fwf(fn("image_list.txt"))
        hq = hq['orig_file']

        splits = pandas.read_csv(fn("list_eval_partition.txt"), delim_whitespace=True, header=None, index_col=0)
        splits = splits.loc[hq]
        
        attr = pandas.read_csv(fn("list_attr_celeba.txt"), delim_whitespace=True, header=1)
        attr = attr.loc[hq]

        mask = np.zeros(len(splits), dtype=bool)
        if test_set == 'original':
            if split == 0:
                # masking true for 64% of the splits
                mask[:int(0.64*len(splits))] = True
            elif split == 1:
                # masking true for 16% of the splits
                mask[int(0.64*len(splits)):int(0.8*len(splits))] = True
            else:
                # masking true for 20% of the splits
                mask[int(0.8*len(splits)):] = True
        else:
            mask[:] = True
        self.filename = splits[mask].index.values
        self.attr = torch.as_tensor(attr[mask].values)

        self.attr = (self.attr + 1) // 2  # map from {-1, 1} to {0, 1}
        self.attr_names = list(attr.columns)
        self.target_idx = self.attr_names.index(self.target_attr)
        self.sensi_idx = self.attr_names.index(self.sensitive_attr)
        self.feature_idx = [i for i in range(len(self.attr_names)) if i != self.target_idx and i!=self.sensi_idx]
        self.num_classes = 2
        self.num_groups =2         
        print('num classes is {}'.format(self.num_classes))

        self.features = [[int(s), int(l), filename] for s, l, filename in \
                            zip(self.attr[:, self.sensi_idx], self.attr[:, self.target_idx], self.filename)]
        self.n_data, self.idxs_per_group = self._data_count(self.features, self.num_groups, self.num_classes)

    def __getitem__(self, index):
        sensitive, target, img_name = self.features[index]
        X = PIL.Image.open(os.path.join(self.root, "img_align_celeba", img_name)).convert('RGB')
        X = ImageOps.fit(X, (256, 256), method=Image.LANCZOS)
        if self.test_pair:
            if self.test_pc_G is not None:
                X_G1 = PIL.Image.open(os.path.join(self.root, self.G1_dir, img_name)).convert('RGB')
                X_edited = PIL.Image.open(os.path.join(self.root, self.ctf_dir, img_name)).convert('RGB')
                X_G2 = PIL.Image.open(os.path.join(self.root, self.G2_dir, img_name)).convert('RGB')
                X =  [X_G1, X_edited, X_G2]
            else:
                X_edited = PIL.Image.open(os.path.join(self.root, self.ctf_dir, img_name)).convert('RGB')
                X =  [X, X_edited]
            target = torch.Tensor([target, target])
            sensitive = torch.Tensor([sensitive, 0 if sensitive ==1 else 1])
        else:
            X = [X]

        if self.transform is not None:
            X = self.transform(X)
            X = torch.stack(X) if (self.test_pair) else X[0]

        return X, 0, sensitive, target, (index, img_name)

    def __len__(self):
        return len(self.features)