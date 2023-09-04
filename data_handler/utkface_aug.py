from os.path import join
from torchvision.datasets.vision import VisionDataset
from PIL import Image
from utils import list_files
from natsort import natsorted
import random
import numpy as np
from data_handler.dataset_factory import GenericDataset
from PIL import ImageOps, Image
import PIL
import os
import torch
from data_handler.utkface import UTKFaceDataset

class UTKFaceDataset_aug(UTKFaceDataset):
    
    def __getitem__(self, index):
        s, l, img_name = self.features[index]
        s = np.float32(s)
        l = np.int64(l)
        # image_path = join(self.root, img_name)
        # image = Image.open(image_path, mode='r').convert('RGB')

        image_path = join(self.root, img_name)
        X = PIL.Image.open(image_path).convert('RGB')
        X = ImageOps.fit(X, (256, 256), method=Image.LANCZOS)
        if self.split == 'train' or self.test_pair:
            # X_edited = PIL.Image.open(os.path.join(self.root, self.base_folder, "img_align_celeba_edited_gender", img_name)).convert('RGB')
            X_edited = PIL.Image.open(os.path.join(self.root+'_edited_gender', img_name)).convert('RGB')
            # why imge fit?
            X =  [X, X_edited]
            l = torch.Tensor([l, l])
            s = torch.Tensor([s, 0 if s ==1 else 1])
        else:
            X = [X]

        if self.transform:
            X = self.transform(X)
            X = torch.stack(X) if (self.split == 'train' or self.test_pair) else X[0]

        return X, 1, s,l, (index, img_name)
    
    def _data_preprocessing(self, filenames):
        filenames = self._delete_incomplete_images(filenames)
        filenames = self._delete_others_n_age_filter(filenames)
        self.features = []
        for filename in filenames:
            s, y = self._filename2SY(filename)
            self.features.append([s, y, filename])


    def _delete_incomplete_images(self, filename):
        filename = [image for image in filename if len(image.split('_')) == 4]
        return filename

    def _delete_others_n_age_filter(self, filename):

        filename = [image for image in filename
                         if ((image.split('_')[self.fea_map['race']] != '4'))]
        ages = [self._transform_age(int(image.split('_')[self.fea_map['age']])) for image in filename]
        self.num_map['age'] = len(set(ages))
        return filename

    def _filename2SY(self, filename):        
        tmp = filename.split('_')
        sensi = int(tmp[self.fea_map[self.sensi]])
        label = int(tmp[self.fea_map[self.label]])
        if self.sensi == 'age':
            sensi = self._transform_age(sensi)
        if self.label == 'age':
            label = self._transform_age(label)
        return int(sensi), int(label)
        
    def _transform_age(self, age):
        if age<20:
            label = 0
        elif age<40:
            label = 1
        else:
            label = 2
        return label 

    def _make_data(self, features, num_groups, num_classes):
        # if the original dataset not is divided into train / test set, this function is used
        min_cnt = 100
        data_count = np.zeros((num_groups, num_classes), dtype=int)
        tmp = []
        for i in reversed(self.features):
            s, l = int(i[0]), int(i[1])
            data_count[s, l] += 1
            if data_count[s, l] <= min_cnt:
                features.remove(i)
                tmp.append(i)

        train_data = features
        test_data = tmp
        return train_data, test_data