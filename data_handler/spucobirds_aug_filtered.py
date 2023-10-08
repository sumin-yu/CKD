import os
import random
import tarfile
from copy import deepcopy
from typing import Callable, Optional

import numpy as np
import requests
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageOps
from tqdm import tqdm
from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple
from torch.utils.data import Dataset
from data_handler.spucobirds import SpuCoBirds, SourceData

TRAIN_SPLIT = "train"
VAL_SPLIT= "valid"
TEST_SPLIT = "test"

# Constants
DOWNLOAD_URL = "https://ucla.box.com/shared/static/zrsx09ik7lqad1e389w06f1c4nbajo9z"
DATASET_NAME = "spuco_birds"
CTF_DATASET_NAME = "spuco_birds_ctf"
LANDBIRDS = "landbirds"
WATERBIRDS = "waterbirds"
LAND = "land"
WATER = "water"
MAJORITY_SIZE = {
    TRAIN_SPLIT: 10000,
    VAL_SPLIT: 500,
    TEST_SPLIT: 500,
}
MINORITY_SIZE = {
    TRAIN_SPLIT: 500,
    VAL_SPLIT: 25,
    TEST_SPLIT: 500,
}

class SpuCoBirds_aug(SpuCoBirds):
    """
    Subset of SpuCoAnimals only including Bird classes.
    """

    def load_data(self) -> SourceData:
        """
        Loads SpuCoBirds and sets spurious labels, label noise.

        :return: The spurious correlation dataset.
        :rtype: SourceData, List[int], List[int]
        """
        
        self.dset_dir = os.path.join(self.root, DATASET_NAME, self.split)
        self.ctf_dset_dir = os.path.join(self.root, CTF_DATASET_NAME, self.split)
        if not os.path.exists(self.dset_dir):
            if not self.download:
                raise RuntimeError(f"Dataset not found {self.dset_dir}, run again with download=True")
            self._download_data()
            self._untar_data()
            os.remove(self.filename)
            
        try:
            self.data = SourceData()
            
            # Landbirds Land 
            landbirds_land = os.listdir(os.path.join(self.dset_dir, f"{LANDBIRDS}/{LAND}"))
            self.data.X.extend([str(os.path.join(self.dset_dir, f"{LANDBIRDS}/{LAND}", x)) for x in landbirds_land])
            if self.split == 'train' or self.test_pair:
                self.data.X_ctf.extend([str(os.path.join(self.ctf_dset_dir, f"{LANDBIRDS}/{LAND}", x)) for x in landbirds_land]) 
            self.data.labels.extend([0] * len(landbirds_land))
            self.data.spurious.extend([0] * len(landbirds_land))
            assert len(landbirds_land) == MAJORITY_SIZE[self.split], f"Dataset corrupted or missing files [landbirds_land]. Expected {MAJORITY_SIZE[self.split]} files got {len(landbirds_land)}"
            
            # Landbirds Water 
            landbirds_water = os.listdir(os.path.join(self.dset_dir, f"{LANDBIRDS}/{WATER}"))
            self.data.X.extend([str(os.path.join(self.dset_dir, f"{LANDBIRDS}/{WATER}", x)) for x in landbirds_water])
            if self.split == 'train' or self.test_pair:
                self.data.X_ctf.extend([str(os.path.join(self.ctf_dset_dir, f"{LANDBIRDS}/{WATER}", x)) for x in landbirds_water])
            self.data.labels.extend([0] * len(landbirds_water))
            self.data.spurious.extend([1] * len(landbirds_water))   
            assert len(landbirds_water) == MINORITY_SIZE[self.split], f"Dataset corrupted or missing files [landbirds_water]. Expected {MINORITY_SIZE[self.split]} files got {len(landbirds_water)}"
            
            # Waterbirds Land
            waterbirds_land = os.listdir(os.path.join(self.dset_dir, f"{WATERBIRDS}/{LAND}"))
            self.data.X.extend([str(os.path.join(self.dset_dir, f"{WATERBIRDS}/{LAND}", x)) for x in waterbirds_land])
            if self.split == 'train' or self.test_pair:
                self.data.X_ctf.extend([str(os.path.join(self.ctf_dset_dir, f"{WATERBIRDS}/{LAND}", x)) for x in waterbirds_land])
            self.data.labels.extend([1] * len(waterbirds_land))
            self.data.spurious.extend([0] * len(waterbirds_land))
            assert len(waterbirds_land) == MINORITY_SIZE[self.split], f"Dataset corrupted or missing files [waterbirds_land]. Expected {MINORITY_SIZE[self.split]} files got {len(waterbirds_land)}"
            
            # Waterbirds Water
            waterbirds_water = os.listdir(os.path.join(self.dset_dir, f"{WATERBIRDS}/{WATER}"))
            self.data.X.extend([str(os.path.join(self.dset_dir, f"{WATERBIRDS}/{WATER}", x)) for x in waterbirds_water])
            if self.split == 'train' or self.test_pair:
                self.data.X_ctf.extend([str(os.path.join(self.ctf_dset_dir, f"{WATERBIRDS}/{WATER}", x)) for x in waterbirds_water])
            self.data.labels.extend([1] * len(waterbirds_water))
            self.data.spurious.extend([1] * len(waterbirds_water)) 
            assert len(waterbirds_water) == MAJORITY_SIZE[self.split], f"Dataset corrupted or missing files [waterbirds_water]. Expected {MAJORITY_SIZE[self.split]} files got {len(waterbirds_water)}"
            
            if self.label_noise > 0.0:
                self.data.clean_labels = deepcopy(self.data.labels)
                self.is_noisy_label = torch.zeros(len(self.data.X))
                self.is_noisy_label[torch.randperm(len(self.data.X))[:int(self.label_noise * len(self.data.X))]] = 1
                self.data.labels = [1 - label if self.is_noisy_label[i] else label for i, label in enumerate(self.data.clean_labels)]
            
        except:
            raise RuntimeError(f"Dataset corrupted, please delete {self.dset_dir} and run again with download=True")
        
        return self.data, list(range(2)), list(range(2))

    def __getitem__(self, index: int):
        """
        Retrieves an item from the dataset.

        :param index: The index of the item.
        :type index: int
        :return: The item at the given index.
        """
        index = self.indices[index]
        X = Image.open(self.data.X[index]).convert('RGB')
        X = ImageOps.fit(X, (256, 256), method=Image.LANCZOS)
        image = [X]
        indicator = 0

        if self.split == 'train' or self.test_pair :    
            if os.path.isfile('./data/spuco/spuco_birds_ctf_filtering_all/' + self.data.X[index].split('/',4)[-1]):
                indicator = 1
            else: indicator = 0
            
        target = self.data.labels[index]
        sensitive = self.spurious[index]
        if self.test_pair or self.split == 'train':
            ctf_image = Image.open(self.data.X_ctf[index]).convert('RGB')
            ctf_image = ImageOps.fit(ctf_image, (256, 256), method=Image.LANCZOS)
            image = [image[0], ctf_image]
            target = torch.Tensor([target, target])
            sensitive = torch.Tensor([sensitive, 0 if sensitive == 1 else 1])
        if self.transform is not None:
            X = self.transform(image)
            X = torch.stack(X) if (self.test_pair or self.split == 'train') else X[0]
        return X, 0, sensitive, target, indicator