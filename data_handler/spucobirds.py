import os
import random
import tarfile
from copy import deepcopy
from typing import Callable, Optional

import numpy as np
import requests
import torch
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm
from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple
from torch.utils.data import Dataset
from data_handler.dataset_factory import GenericDataset

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

class SourceData():
    """
    Class representing the source data.

    This class contains the input data and corresponding labels.
    """

    def __init__(self, data=None):
        """
        Initialize the SourceData object.

        :param data: The input data and labels.
        :type data: List[Tuple]
        """
        self.X = []
        self.X_ctf = []
        self.labels = []
        self.spurious = []
        self.clean_labels = None
        self.core_feature_noise = None
        if data is not None:
            for x, label in tqdm(data):
                self.X.append(x)
                self.X_ctf.append(x)
                self.labels.append(label)

class BaseSpuCoDataset(Dataset):
    def __init__(
        self,
        root: str,
        num_classes: int,
        split: str = "train",
        transform: Optional[Callable] = None,
        verbose: bool = False,
    ):
        """
        Initializes the dataset.

        :param root: Root directory of the dataset.
        :type root: str

        :param num_classes: Number of classes in the dataset.
        :type num_classes: int

        :param split: Split of the dataset (e.g., "train", "test", "val"). Defaults to "train".
        :type split: str, optional

        :param transform: Optional transform to be applied to the data. Defaults to None.
        :type transform: Callable, optional

        :param verbose: Whether to print verbose information during dataset initialization. Defaults to False.
        :type verbose: bool, optional
        """
        
        super().__init__(root=root)
        self.root = root 
        self._num_classes = num_classes
        assert split == TRAIN_SPLIT or split == VAL_SPLIT or split == TEST_SPLIT, f"split must be one of {TRAIN_SPLIT}, {VAL_SPLIT}, {TEST_SPLIT}"
        self.split = split
        self.transform = transform
        self.verbose = verbose
        self.skip_group_validation = False

    def initialize(self):
        """
        Initializes the dataset.
        """
        # Load Data
        self.data, classes, spurious_classes = self.load_data()
        self.num_spurious = len(spurious_classes)
        
        # Group Partition
        self._group_partition = {}
        for i, group_label in enumerate(zip(self.data.labels, self.spurious)):
            if group_label not in self._group_partition:
                self._group_partition[group_label] = []
            self._group_partition[group_label].append(i)

        self._clean_group_partition = None
        if self.data.clean_labels is not None:
            self._clean_group_partition = {}
            for i, group_label in enumerate(zip(self.data.clean_labels, self.spurious)):
                if group_label not in self._clean_group_partition:
                    self._clean_group_partition[group_label] = []
                self._clean_group_partition[group_label].append(i)
            
        # Validate partition sizes
        if not self.skip_group_validation:
            for class_label in classes:
                for spurious_label in spurious_classes:
                    group_label = (class_label, spurious_label)
                    assert group_label in self._group_partition and len(self._group_partition[group_label]) > 0, f"No examples in {group_label}, considering reducing spurious correlation strength"

        # Group Weights
        self._group_weights = {}
        for key in self._group_partition.keys():
            self._group_weights[key] = len(self._group_partition[key]) / len(self.data.X)
        
        self.indices = range(len(self.data.X))
            
                
    @property
    def group_partition(self) -> Dict[Tuple[int, int], List[int]]:
        """
        Dictionary partitioning indices into groups

        :rtype: Dict[Tuple[int, int], List[int]]
        """
        return self._group_partition


    @property
    def clean_group_partition(self) -> Dict[Tuple[int, int], List[int]]:
        """
        Dictionary partitioning indices into groups based on clean labels

        :rtype: Dict[Tuple[int, int], List[int]]
        """
        if self._clean_group_partition is None:
            return self._group_partition
        else:
            return self._clean_group_partition


    @property
    def group_weights(self) -> Dict[Tuple[int, int], float]:
        """
        Dictionary containing the fractional weights of each group

        :rtype: Dict[Tuple[int, int], float]
        """
        return self._group_weights


    @property
    def spurious(self) -> List[int]:
        """
        List containing spurious labels for each example

        :rtype: List[int]
        """
        return self.data.spurious


    @property
    def labels(self) -> List[int]:
        """
        List containing class labels for each example

        :rtype: List[int]
        """
        return self.data.labels


    @property
    def num_classes(self) -> int:
        """
        Number of classes

        :rtype: int
        """
        return self._num_classes

        
    def __getitem__(self, index):
        """
        Gets an item from the dataset.

        :param index: Index of the item to get.
        :type index: int
        :return: A tuple of (sample, target) where target is class_index of the target class.
        :rtype: tuple
        """
        if self.transform is None:
            return self.data.X[index], self.data.labels[index]
        else:
            return self.transform(self.data.X[index]), self.data.labels[index]
        
    def __len__(self):
        """
        Gets the length of the dataset.

        :return: Length of the dataset.
        :rtype: int
        """
        return len(self.data.X)
    
class SpuCoBirds(BaseSpuCoDataset, GenericDataset):
    """
    Subset of SpuCoAnimals only including Bird classes.
    """

    def __init__(
        self,
        root: str,
        download: bool = True,
        label_noise: float = 0.0,
        split: str = TRAIN_SPLIT,
        transform: Optional[Callable] = None,
        verbose: bool = False,
        num_aug: int = 1
    ):
        """
        Initializes the dataset.

        :param root: Root directory of the dataset.
        :type root: str

        :param download: Whether to download the dataset.
        :type download: bool, optional

        :param label_noise: The amount of label noise to apply.
        :type label_noise: float, optional

        :param split: The split of the dataset.
        :type split: str, optional

        :param transform: Optional transform to be applied to the data.
        :type transform: Callable, optional

        :param verbose: Whether to print verbose information during dataset initialization.
        :type verbose: bool, optional
        """

        super().__init__(
            root=root, 
            split=split,
            transform=transform,
            num_classes=2,
            verbose=verbose
        )
        self.download = download
        self.label_noise = label_noise
        self.num_groups = 2
        self.test_pair = False
        self.num_aug = num_aug

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
            if self.test_pair:
                self.data.X_ctf.extend([str(os.path.join(self.ctf_dset_dir, f"{LANDBIRDS}/{LAND}", x)) for x in landbirds_land]) 
            self.data.labels.extend([0] * len(landbirds_land))
            self.data.spurious.extend([0] * len(landbirds_land))
            assert len(landbirds_land) == MAJORITY_SIZE[self.split], f"Dataset corrupted or missing files [landbirds_land]. Expected {MAJORITY_SIZE[self.split]} files got {len(landbirds_land)}"
            
            # Landbirds Water 
            landbirds_water = os.listdir(os.path.join(self.dset_dir, f"{LANDBIRDS}/{WATER}"))
            self.data.X.extend([str(os.path.join(self.dset_dir, f"{LANDBIRDS}/{WATER}", x)) for x in landbirds_water])
            if self.test_pair:
                self.data.X_ctf.extend([str(os.path.join(self.ctf_dset_dir, f"{LANDBIRDS}/{WATER}", x)) for x in landbirds_water])
            self.data.labels.extend([0] * len(landbirds_water))
            self.data.spurious.extend([1] * len(landbirds_water))   
            assert len(landbirds_water) == MINORITY_SIZE[self.split], f"Dataset corrupted or missing files [landbirds_water]. Expected {MINORITY_SIZE[self.split]} files got {len(landbirds_water)}"
            
            # Waterbirds Land
            waterbirds_land = os.listdir(os.path.join(self.dset_dir, f"{WATERBIRDS}/{LAND}"))
            self.data.X.extend([str(os.path.join(self.dset_dir, f"{WATERBIRDS}/{LAND}", x)) for x in waterbirds_land])
            if self.test_pair:
                self.data.X_ctf.extend([str(os.path.join(self.ctf_dset_dir, f"{WATERBIRDS}/{LAND}", x)) for x in waterbirds_land])
            self.data.labels.extend([1] * len(waterbirds_land))
            self.data.spurious.extend([0] * len(waterbirds_land))
            assert len(waterbirds_land) == MINORITY_SIZE[self.split], f"Dataset corrupted or missing files [waterbirds_land]. Expected {MINORITY_SIZE[self.split]} files got {len(waterbirds_land)}"
            
            # Waterbirds Water
            waterbirds_water = os.listdir(os.path.join(self.dset_dir, f"{WATERBIRDS}/{WATER}"))
            self.data.X.extend([str(os.path.join(self.dset_dir, f"{WATERBIRDS}/{WATER}", x)) for x in waterbirds_water])
            if self.test_pair:
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

    def _download_data(self):
        self.filename = f"{self.root}/{DATASET_NAME}.tar.gz"

        response = requests.get(DOWNLOAD_URL, stream=True)
        response.raise_for_status()

        with open(self.filename, "wb") as file:
            for chunk in tqdm(response.iter_content(chunk_size=1024), total=2952065, desc="Downloading SpuCoBirds", unit="KB"):
                file.write(chunk)
    
    def _untar_data(self):
        # Open the tar.gz file
        with tarfile.open(self.filename, "r:gz") as tar:
            # Extract all files to the specified output directory
            tar.extractall(self.root)
            
    def __getitem__(self, index):
        """
        Gets an item from the dataset.

        :param index: Index of the item to get.
        :type index: int
        :return: A tuple of (sample, target) where target is class_index of the target class.
        :rtype: tuple
        """

    def __getitem__(self, index: int):
        """
        Retrieves an item from the dataset.

        :param index: The index of the item.
        :type index: int
        :return: The item at the given index.
        """
        index = self.indices[index]
        image = [Image.open(self.data.X[index]).convert('RGB')]
        target = self.data.labels[index]
        sensitive = self.spurious[index]
        if self.test_pair:
            ctf_image = Image.open(self.data.X_ctf[index]).convert('RGB')
            image = [image[0], ctf_image]
            target = torch.Tensor([target, target])
            sensitive = torch.Tensor([sensitive, 0 if sensitive == 1 else 1])
        if self.transform is not None:
            X = self.transform(image)
            X = torch.stack(X) if (self.test_pair) else X[0]
        return X, 0, sensitive, target, 0