import torch
import os
from os.path import join
import PIL
from PIL import ImageOps, Image
from data_handler.dataset_factory import GenericDataset
from data_handler.celeba_hq import CelebA_HQ

class CelebA_HQ_aug(CelebA_HQ):
    def __getitem__(self, index):
        img_name = self.filename[index]
        X = PIL.Image.open(os.path.join(self.root, "img_align_celeba", img_name)).convert('RGB')
        X = ImageOps.fit(X, (256, 256), method=Image.LANCZOS)
        if self.split == 'train':
            X_edited = PIL.Image.open(os.path.join(self.root, self.ctf_dir, img_name)).convert('RGB')
            X =  [X, X_edited]
            target = self.attr[index, self.target_idx]
            target = torch.Tensor([target, target])
            sensitive = self.attr[index, self.sensi_idx]
            sensitive = torch.Tensor([sensitive, 0 if sensitive == 1 else 1])
        elif self.test_pair:
            if self.test_pc_G is not None:
                X_G1 = PIL.Image.open(os.path.join(self.root, self.G1_dir, img_name)).convert('RGB')
                X_edited = PIL.Image.open(os.path.join(self.root, self.ctf_dir, img_name)).convert('RGB')
                X_G2 = PIL.Image.open(os.path.join(self.root, self.G2_dir, img_name)).convert('RGB')
                X =  [X_G1, X_edited, X_G2]
            else:
                X_edited = PIL.Image.open(os.path.join(self.root, self.ctf_dir, img_name)).convert('RGB')
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
                
        return X, 0, sensitive, target, (index, img_name)

    def __len__(self):
        return len(self.attr)