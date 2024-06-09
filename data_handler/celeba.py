import torch
import os
from os.path import join
import PIL
from PIL import ImageOps, Image
import pandas
import numpy as np
import zipfile
from functools import partial
from torchvision.datasets.utils import download_file_from_google_drive, check_integrity, verify_str_arg
from data_handler.dataset_factory import GenericDataset

class CelebA(GenericDataset):
    base_folder = "celeba"
    # There currently does not appear to be a easy way to extract 7z in python (without introducing additional
    # dependencies). The "in-the-wild" (not aligned+cropped) images are only in 7z, so they are not available
    # right now.
    file_list = [
        # File ID                         MD5 Hash                            Filename
        ("0B7EVK8r0v71pZjFTYXZWM3FlRnM", "00d2c5bc6d35e252742224ab0c1e8fcb", "img_align_celeba.zip"),
        # ("0B7EVK8r0v71pbWNEUjJKdDQ3dGc", "b6cd7e93bc7a96c2dc33f819aa3ac651", "img_align_celeba_png.7z"),
        # ("0B7EVK8r0v71peklHb0pGdDl6R28", "b6cd7e93bc7a96c2dc33f819aa3ac651", "img_celeba.7z"),
        ("0B7EVK8r0v71pblRyaVFSWGxPY0U", "75e246fa4810816ffd6ee81facbd244c", "list_attr_celeba.txt"),
        ("1_ee_0u7vcNLOfNLegJRHmolfH5ICW-XS", "32bd1bd63d3c78cd57e08160ec5ed1e2", "identity_CelebA.txt"),
        ("0B7EVK8r0v71pbThiMVRxWXZ4dU0", "00566efa6fedff7a56946cd1c10f1c16", "list_bbox_celeba.txt"),
        ("0B7EVK8r0v71pd0FJY3Blby1HUTQ", "cc24ecafdb5b50baae59b03474781f8c", "list_landmarks_align_celeba.txt"),
        # ("0B7EVK8r0v71pTzJIdlJWdHczRlU", "063ee6ddb681f96bc9ca28c6febb9d1a", "list_landmarks_celeba.txt"),
        ("0B7EVK8r0v71pY0NSMzRuSXJEVkk", "d32c9cbf5e040fd4025c592c306e6668", "list_eval_partition.txt"),
    ]

    def __init__(self, root, split="train", target_type="attr", transform=None,
                 target_transform=None, download=False, target_attr='Blond_Hair', sen_attr='Male', test_set='original', test_pc_G=None):
        super(CelebA, self).__init__(root, transform=transform)
        self.split = split
        self.test_pair = False
        self.ctf_dir = "img_align_celeba_edited_{}".format(sen_attr)
        self.test_pc_G = test_pc_G
        if test_pc_G is not None:
            if test_pc_G == 'Hair_Length':
                self.G1_dir = "img_align_celeba_Hair_Length_ctf_with_origin"
                self.G2_dir = "img_align_celeba_Hair_Length_org_with_origin"


        if isinstance(target_type, list):
            self.target_type = target_type
        else:
            self.target_type = [target_type]

        if not self.target_type and self.target_transform is not None:
            raise RuntimeError('target_transform is specified but target_type is empty')

        if download:
            self.download()
        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')
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

        fn = partial(join, self.root, self.base_folder)
        splits = pandas.read_csv(fn("list_eval_partition.txt"), delim_whitespace=True, header=None, index_col=0)
        if test_set == 'pc_G':
            attr = pandas.read_csv(fn("list_attr_celeba_test_strong_filter_{}_{}_pc_G_{}.txt".format(self.target_attr, self.sensitive_attr, self.test_pc_G)))
            self.attr = torch.as_tensor(attr.values)
            self.filename = attr.index.values
        elif test_set == 'original':
            attr = pandas.read_csv(fn("list_attr_celeba.txt"), delim_whitespace=True, header=1)
            mask = slice(None) if split is None else (splits[1] == split)
            self.filename = splits[mask].index.values
            self.attr = torch.as_tensor(attr[mask].values)
        elif test_set == 'strong_f':
            attr = pandas.read_csv(fn("list_attr_celeba_test_strong_filter_{}_{}.txt".format(self.target_attr, self.sensitive_attr)))
            # attr = pandas.read_csv(fn("list_attr_celeba_test_strong_filter_{}_{}_pc_G_{}.txt".format(self.target_attr, self.sensitive_attr, 'Hair_Length')))
            self.attr = torch.as_tensor(attr.values)
            self.filename = attr.index.values

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
        
    def _check_integrity(self):
        for (_, md5, filename) in self.file_list:
            fpath = os.path.join(self.root, self.base_folder, filename)
            _, ext = os.path.splitext(filename)
            # Allow original archive to be deleted (zip and 7z)
            # Only need the extracted images
            if ext not in [".zip", ".7z"] and not check_integrity(fpath, md5):
                return False

        # Should check a hash of the images
        return os.path.isdir(os.path.join(self.root, self.base_folder, "img_align_celeba"))

    def download(self):
        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        for (file_id, md5, filename) in self.file_list:
            download_file_from_google_drive(file_id, os.path.join(self.root, self.base_folder), filename, md5)

        with zipfile.ZipFile(os.path.join(self.root, self.base_folder, "img_align_celeba.zip"), "r") as f:
            f.extractall(os.path.join(self.root, self.base_folder))

    def __getitem__(self, index):
        sensitive, target, img_name = self.features[index]
        X = PIL.Image.open(os.path.join(self.root, self.base_folder, "img_align_celeba", img_name)).convert('RGB')
        X = ImageOps.fit(X, (256, 256), method=Image.LANCZOS)
        if self.test_pair:
            if self.test_pc_G is not None:
                X_G1 = PIL.Image.open(os.path.join(self.root, self.base_folder, self.G1_dir, img_name)).convert('RGB')
                X_G1 = ImageOps.fit(X_G1, (256, 256), method=Image.LANCZOS)
                X_G2 = PIL.Image.open(os.path.join(self.root, self.base_folder, self.G2_dir, img_name)).convert('RGB')
                X_G2 = ImageOps.fit(X_G2, (256, 256), method=Image.LANCZOS)
                X =  [X_G1, X_G2]
            else:
                X = PIL.Image.open(os.path.join(self.root, self.base_folder, "img_align_celeba_edited_original", img_name)).convert('RGB')
                X = ImageOps.fit(X, (256, 256), method=Image.LANCZOS)
                X_edited = PIL.Image.open(os.path.join(self.root, self.base_folder, self.ctf_dir, img_name)).convert('RGB')
                X_edited = ImageOps.fit(X_edited, (256, 256), method=Image.LANCZOS)
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