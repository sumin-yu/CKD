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
from data_handler.dataset_factory import GenericDataset

class RAF(GenericDataset):
    label = 'emotion'
    sensi = 'gender'
    
    def __init__(self, root, split='train', transform=None, target_transform=None, num_aug=1, img_cfg=2.0):
        super(RAF, self).__init__(root, transform=transform)
        self.root = root # './data/rafdb'
        self.org_dir = 'Image/aligned'
        self.ctf_dir = 'Image/aligned_edited_gender'
        self.test_ctf_dir = 'Image/aligned_edited_gender_testcfg2.0' if img_cfg==2.0 else 'Image/aligned_edited_gender_testcfg1.8'
        self.sensi_dir = 'Annotation/manual'
        self.split = split
        self.num_aug = num_aug
        self.num_groups = 2
        self.num_classes = 7
        self.test_pair = False

        fn = partial(join, self.root, 'EmoLabel')
        self.attr = pandas.read_csv(fn("list_patition_label.txt"), header=None, sep=' ', names=['img_name','tar'])
        self.attr['tar'] = self.attr['tar'].apply(lambda x : x-1) # change label 1~7 to 0~6

        # make test or train set
        split_set = self.split if self.split != 'valid' else 'test'
        self.attr = self.attr[self.attr['img_name'].str.contains(split_set)]

        # make validation set
        # if self.split != 'test':
        #     self.attr = self.attr[:int(len(self.attr)/5)] if (self.split != 'train') else self.attr[int(len(self.attr)/5):]

        self.img_list = self.attr['img_name'].values.tolist()

        # add group label to dataframe
        self._annotate_sensi()

        # remove unsure gender
        unsure_idx = self.attr[self.attr['sensi']==2].index
        self.attr = self.attr.drop(unsure_idx)

        self.attr = self.attr.reset_index()

        self.features = [[int(s), int(l), img_name] for img_name, l, s in zip(self.attr['img_name'], self.attr['tar'], self.attr['sensi'])]
        self.n_data, self.idxs_per_group = self._data_count(self.features, self.num_groups, self.num_classes)

    def __len__(self):
        return len(self.attr)
    
    def _annotate_sensi(self):
        sensi_list = []
        img_list = self.attr['img_name'].tolist()
        fn = partial(join, self.root, self.sensi_dir)
        for img in img_list:
            filename = img.replace('.jpg','_manu_attri.txt')
            with open(fn(filename), 'r') as f:
                sensi = f.readlines()[5][0]
                align_sensi = 0 if sensi == '1'  else 1
                sensi_list.append(align_sensi)
        self.attr['sensi'] = sensi_list

    def __getitem__(self, index):
        img_name = self.img_list[index]
        img_name_ = img_name.replace('.jpg', '_aligned.jpg')

        X = PIL.Image.open(os.path.join(self.root, self.org_dir, img_name_)).convert('RGB')
        if self.test_pair:
            X_edited = PIL.Image.open(os.path.join(self.root, self.test_ctf_dir, img_name_)).convert('RGB')
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
            X = self.transform(X)
            X = torch.stack(X) if (self.test_pair) else X[0]
        
        return X, 0, sensitive, target, (index, img_name)