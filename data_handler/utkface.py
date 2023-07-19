from os.path import join
from torchvision.datasets.vision import VisionDataset
from PIL import Image
from utils import list_files
from natsort import natsorted
import random
import numpy as np
from data_handler.dataset_factory import GenericDataset

class UTKFaceDataset(GenericDataset):
    
    label = 'age'
    sensi = 'gender'
    fea_map = {
        'age' : 0,
        'gender' : 1,
        'race' : 2
    }
    num_map = {
        'age' : 100,
        'gender' : 2,
        'race' : 4
    }

    def __init__(self, root, split='train', transform=None, target_transform=None):
        
        super(UTKFaceDataset, self).__init__(root, transform=transform)
        
        self.split = split
        self.filename = list_files(root, '.jpg')
        self.filename = natsorted(self.filename)

        self.num_groups = self.num_map[self.sensi]
        self.num_classes = self.num_map[self.label]        

        random.seed(1)
        random.shuffle(self.features)

        train, test = self._make_data(self.features, self.num_groups, self.num_classes)
        self.features = train if self.split == 'train' or 'group' in self.version else test

        self.n_data, self.idxs_per_group = self._data_count(self.features, self.num_groups, self.num_classes)        

    def __len__(self):
        return len(self.filename)

    def __getitem__(self, index):
        s, l, img_name = self.features[index]

        image_path = join(self.root, img_name)
        image = Image.open(image_path, mode='r').convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, 1, np.float32(s), np.int64(l), (index, img_name)
    
    def _data_preprocessing(self, filenames):
        filenames = self._delete_incomplete_images(filenames)
        filenames = self._delete_others_n_age_filter(filenames)
        self.features = []
        for filename in filenames:
            s, y = self._filename2SY(filename)
            self.features.append([s, y, filename])


    def _delete_incomplete_images(self):
        self.filename = [image for image in self.filename if len(image.split('_')) == 4]

    def _delete_others_n_age_filter(self):

        self.filename = [image for image in self.filename
                         if ((image.split('_')[self.fea_map['race']] != '4'))]
        ages = [self._transform_age(int(image.split('_')[self.fea_map['age']])) for image in self.filename]
        self.num_map['age'] = len(set(ages))

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