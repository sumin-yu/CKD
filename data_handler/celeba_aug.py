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
        if self.split == 'train':
            X_edited = PIL.Image.open(os.path.join(self.root, self.base_folder, self.ctf_dir, img_name)).convert('RGB')
            X =  [X, X_edited]
            target = self.attr[index, self.target_idx]
            target = torch.Tensor([target, target])
            sensitive = self.attr[index, self.sensi_idx]
            sensitive = torch.Tensor([sensitive, 0 if sensitive == 1 else 1])
        if self.test_pair:
            if self.test_pc_G is not None:
                X_G1 = PIL.Image.open(os.path.join(self.root, self.base_folder, self.G1_dir, img_name)).convert('RGB')
                X_G1 = ImageOps.fit(X_G1, (256, 256), method=Image.LANCZOS)
                X_G2 = PIL.Image.open(os.path.join(self.root, self.base_folder, self.G2_dir, img_name)).convert('RGB')
                X_G2 = ImageOps.fit(X_G2, (256, 256), method=Image.LANCZOS)
                X_G1.save('X_G1_celeba.png')
                X_G2.save('X_G2_celeba.png')
                X =  [X_G1, X_G2]
            else:
                X_edited = PIL.Image.open(os.path.join(self.root, self.base_folder, self.ctf_dir, img_name)).convert('RGB')
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
            if self.split == 'train':
                if self.num_aug == 1:
                    X = self.transform(X)
                    X = torch.stack(X) 
                else:
                    X_list = []
                    for i in range(self.num_aug):
                        tmp = self.transform(X)
                        X_list.extend(tmp)
                    X = torch.stack(X_list) 
                    target = target.repeat(self.num_aug)
                    sensitive = sensitive.repeat(self.num_aug)
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
