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
    def __getitem__(self, index):
        img_name = self.filename[index]
        if self.split == 'train':
            X = PIL.Image.open(os.path.join(self.root, self.base_folder, "img_align_celeba", img_name)).convert('RGB')
            X = ImageOps.fit(X, (256, 256), method=Image.LANCZOS)
            X_edited = PIL.Image.open(os.path.join(self.root, self.base_folder, self.ctf_dir, img_name)).convert('RGB')
            X =  [X, X_edited]
            target = self.attr[index, self.target_idx]
            target = torch.Tensor([target, target])
            sensitive = self.attr[index, self.sensi_idx]
            sensitive = torch.Tensor([sensitive, 0 if sensitive == 1 else 1])
        elif self.test_pair:
            X = PIL.Image.open(os.path.join(self.root, self.base_folder, "img_align_celeba_edited_original", img_name)).convert('RGB')
            X = ImageOps.fit(X, (256, 256), method=Image.LANCZOS)
            X_edited = PIL.Image.open(os.path.join(self.root, self.base_folder, self.ctf_dir, img_name)).convert('RGB')
            X_edited = ImageOps.fit(X_edited, (256, 256), method=Image.LANCZOS)
            X =  [X, X_edited]
            target = self.attr[index, self.target_idx]
            target = torch.Tensor([target, target])
            sensitive = self.attr[index, self.sensi_idx]
            sensitive = torch.Tensor([sensitive, 0 if sensitive == 1 else 1])
        else:
            X = PIL.Image.open(os.path.join(self.root, self.base_folder, "img_align_celeba", img_name)).convert('RGB')
            X = ImageOps.fit(X, (256, 256), method=Image.LANCZOS)
            X = [X]
            target = self.attr[index, self.target_idx]
            sensitive = self.attr[index, self.sensi_idx]

        feature = self.attr[index, self.feature_idx]
        if self.transform is not None:
            if self.split == 'train':
                X = self.transform(X)
                X = torch.stack(X) 
            else:
                if self.test_pair:
                    X = self.transform(X)
                    X = torch.stack(X) 
                else:
                    X = self.transform(X)
                    X = X[0]
                
        return X, feature, sensitive, target, (index, img_name)

    def __len__(self):
        return len(self.attr)