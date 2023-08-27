import torch
import os
from os.path import join
import PIL
from PIL import ImageOps, Image
from data_handler.celeba import CelebA

class CelebA_aug(CelebA):
    def __getitem__(self, index):
        img_name = self.filename[index]
        X = PIL.Image.open(os.path.join(self.root, self.base_folder, "img_align_celeba", img_name)).convert('RGB')
        X = ImageOps.fit(X, (256, 256), method=Image.LANCZOS)
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
            tmp = []
            num_transform = self.num_aug if self.split == 'train' else 1
            for _ in range(num_transform):
                    X = self.transform(X)
                    tmp.extend(X)
            X = torch.stack(tmp) if (self.split == 'train' or self.test_pair) else X[0]
            
        return X, feature, sensitive, target, (index, img_name)

    def __len__(self):
        return len(self.attr)
