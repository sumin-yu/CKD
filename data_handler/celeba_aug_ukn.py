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
from data_handler.celeba import CelebA

class CelebA_aug(CelebA):
    def __init__(self, root, split="train", target_type="attr", transform=None, target_transform=None, download=False, target_attr='Blond_Hair', sen_attr='Male', method='kd_mfd_indiv'):
        super().__init__(root, split, target_type, transform, target_transform, download, target_attr, sen_attr)
        self.method = method

    def __getitem__(self, index):
        img_name = self.filename[index]
        if self.split == 'train':
            X = PIL.Image.open(os.path.join(self.root, self.base_folder, "img_align_celeba", img_name)).convert('RGB')
            X = ImageOps.fit(X, (256, 256), method=Image.LANCZOS)
            if self.method == 'kd_indiv' :
                sensitive = self.attr[index, self.sensi_idx]
                X_edited = PIL.Image.open(os.path.join(self.root, self.base_folder, "img_align_celeba_edited_gender_ukn_m", img_name)).convert('RGB') if sensitive == 0 else PIL.Image.open(os.path.join(self.root, self.base_folder, "img_align_celeba_edited_gender_ukn_w", img_name)).convert('RGB')
                sensitive = torch.Tensor([sensitive, 0 if sensitive == 1 else 1])
                target = self.attr[index, self.target_idx]
                target = torch.Tensor([target, target])
                X = [X, X_edited]
            elif self.method == 'scratch':
                sensitive = self.attr[index, self.sensi_idx]
                sensitive = 0 if sensitive == 1 else 1
                X_edited = PIL.Image.open(os.path.join(self.root, self.base_folder, "img_align_celeba_edited_gender_ukn_m", img_name)).convert('RGB') if sensitive == 0 else PIL.Image.open(os.path.join(self.root, self.base_folder, "img_align_celeba_edited_gender_ukn_w", img_name)).convert('RGB')
                target = self.attr[index, self.target_idx]
                X_edited = ImageOps.fit(X_edited, (256, 256), method=Image.LANCZOS)
                X = [X_edited]
            else:
                X_edited1 = PIL.Image.open(os.path.join(self.root, self.base_folder, "img_align_celeba_edited_gender_ukn_m", img_name)).convert('RGB')
                X_edited2 = PIL.Image.open(os.path.join(self.root, self.base_folder, "img_align_celeba_edited_gender_ukn_w", img_name)).convert('RGB')
                X_edited1 = ImageOps.fit(X_edited1, (256, 256), method=Image.LANCZOS)
                X_edited2 = ImageOps.fit(X_edited2, (256, 256), method=Image.LANCZOS)
                X = [X, X_edited1, X_edited2]
                sensitive = self.attr[index, self.sensi_idx]
                sensitive = torch.Tensor([sensitive, 1, 0])
                target = self.attr[index, self.target_idx]
                target = torch.Tensor([target, target, target])
        elif self.test_pair:
            X = PIL.Image.open(os.path.join(self.root, self.base_folder, "img_align_celeba", img_name)).convert('RGB')
            X = ImageOps.fit(X, (256, 256), method=Image.LANCZOS)
            X_edited = PIL.Image.open(os.path.join(self.root, self.base_folder, "img_align_celeba_edited_gender", img_name)).convert('RGB')
            X_edited = ImageOps.fit(X_edited, (256, 256), method=Image.LANCZOS)
            X = [X, X_edited] 
            sensitive = self.attr[index, self.sensi_idx]
            sensitive = torch.Tensor([sensitive, 0 if sensitive == 1 else 1])
            target = self.attr[index, self.target_idx]
            target = torch.Tensor([target, target])
        else: 
            X = PIL.Image.open(os.path.join(self.root, self.base_folder, "img_align_celeba", img_name)).convert('RGB')
            X = ImageOps.fit(X, (256, 256), method=Image.LANCZOS)
            X = [X]
            sensitive = self.attr[index, self.sensi_idx]
            target = self.attr[index, self.target_idx]

        feature = self.attr[index, self.feature_idx]

        if self.transform is not None:
            X = self.transform(X)
            X = torch.stack(X) if (self.split == 'train' or self.test_pair) and (self.method != 'scratch') else X[0]
            
        return X, feature, sensitive, target, (index, img_name)

    def __len__(self):
        return len(self.attr)