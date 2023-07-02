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
        X = PIL.Image.open(os.path.join(self.root, self.base_folder, "img_align_celeba", img_name)).convert('RGB')
        X = ImageOps.fit(X, (256, 256), method=Image.LANCZOS)
        if self.split == 'train' or self.test_pair:
            if self.sensitive_attr == 'Male':
                X_edited = PIL.Image.open(os.path.join(self.root, self.base_folder, "img_align_celeba_edited_gender", img_name)).convert('RGB')
            elif self.sensitive_attr == 'Blond_Hair':
                X_edited = PIL.Image.open(os.path.join(self.root, self.base_folder, "img_align_celeba_edited", img_name)).convert('RGB')
            X =  [X, X_edited]
        else:
            X = [X]

        target = self.attr[index, self.target_idx]
        sensitive = self.attr[index, self.sensi_idx]
        feature = self.attr[index, self.feature_idx]

        if self.transform is not None:
            X = self.transform(X)
            X = torch.stack(X) if (self.split == 'train' or self.test_pair) else X[0]
            
        return X, feature, sensitive, target, (index, img_name)

    def __len__(self):
        return len(self.attr)

    # def _balance_test_data(self):
    #     num_data_min = np.min(self.num_data)
    #     print('min : ', num_data_min)
    #     data_count = np.zeros((self.num_groups, self.num_classes), dtype=int)
    #     new_filename = []
    #     new_attr = []
    #     print(len(self.attr))        
    #     for index in range(len(self.attr)):
    #         target=self.attr[index, self.target_idx]
    #         sensitive = self.attr[index, self.sensi_idx]
    #         if data_count[sensitive, target] < num_data_min:
    #             new_filename.append(self.filename[index])
    #             new_attr.append(self.attr[index])
    #             data_count[sensitive, target] += 1
            
    #     for i in range(self.num_groups):
    #         print('# of balanced %d\'s groups data : '%i, data_count[i, :])
            
    #     self.filename = new_filename
    #     self.attr = torch.stack(new_attr)
