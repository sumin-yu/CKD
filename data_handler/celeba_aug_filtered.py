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
import pickle

class CelebA_aug(CelebA):
    def __getitem__(self, index):
        img_name = self.filename[index]
        X = PIL.Image.open(os.path.join(self.root, self.base_folder, "img_align_celeba", img_name)).convert('RGB')
        X = ImageOps.fit(X, (256, 256), method=Image.LANCZOS)

        indicator = 0
        if self.split == 'train':
            file_name = "img_align_celeba_edited_gender_filtered_" + self.target_attr
            if os.path.isfile(os.path.join(self.root, self.base_folder, file_name, img_name)):
                indicator = 1
            else: indicator = 0
        if self.split == 'train' or self.test_pair:
            X_edited = PIL.Image.open(os.path.join(self.root, self.base_folder, "img_align_celeba_edited_gender", img_name)).convert('RGB')
            X =  [X, X_edited]
            target = self.attr[index, self.target_idx]
            target = torch.Tensor([target, target])
            sensitive = self.attr[index, self.sensi_idx]
            sensitive = torch.Tensor([sensitive, 0 if sensitive == 1 else 1])
        else:
            X = [X]
            target = self.attr[index, self.target_idx]
            sensitive = self.attr[index, self.sensi_idx]

        feature = self.attr[index, self.feature_idx]

        if self.transform is not None:
            X = self.transform(X)
            X = torch.stack(X) if (self.split == 'train' or self.test_pair) else X[0]
            
        return X, feature, sensitive, target, indicator
    
    def __len__(self):
        return len(self.attr)