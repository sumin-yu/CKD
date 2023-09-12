import torch
import os
from os.path import join
import PIL
from PIL import ImageOps, Image
import pandas
from functools import partial
from utils import list_files
from natsort import natsorted
import numpy as np
from data_handler.raf import RAF

class RAF_aug(RAF):
    def __getitem__(self, index):
        img_name = self.img_list[index]
        img_name_ = img_name.replace('.jpg', '_aligned.jpg')

        X = PIL.Image.open(os.path.join(self.root, self.org_dir, img_name_)).convert('RGB')
        if self.split =='train' or self.test_pair:
            ctf_img_dir = self.test_ctf_dir if self.test_pair else self.ctf_dir
            X_edited = PIL.Image.open(os.path.join(self.root, ctf_img_dir, img_name_)).convert('RGB')
            X = [X, X_edited]
            target = self.attr['tar'][index]
            target = torch.Tensor([target, target])
            sensitive = self.attr['sensi'][index]
            sensitive = torch.Tensor([sensitive, 0 if sensitive == 1 else 1])
        else:
            X = [X]
            target = self.attr['tar'][index]
            sensitive = self.attr['sensi'][index]
        
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
        
        return X, 0, sensitive, target, (index, img_name)