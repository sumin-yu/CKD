import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from PIL import Image

from torchvision.datasets.utils import check_integrity, download_and_extract_archive, download_url, verify_str_arg
from data_handler.dataset_factory import GenericDataset

import pandas as pd
from functools import partial
from os.path import join

import torch
from data_handler.lfw import LFWPeople

class LFWPeople_aug(LFWPeople):
    def __init__(
        self,
        root: str,
        split: str = "10fold",
        image_set: str = "funneled",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = True, 
        target_attr: str = "Blond_Hair",
        sen_attr: str = "Male",

    ) -> None:
        super().__init__(root, split, image_set, transform, target_transform, download, target_attr, sen_attr)

        self.images_ctf_dir = join(self.root, self.base_folder, "lfw_funneled_Male")
        self.ctf_data = self._get_people_ctf()

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        sensitive, target, img_name = self.features[index]
        img = self._loader(self.data[index])
        img_edited = self._loader(self.ctf_data[index])
        if self.test_pair or self.split == 'train':
            img_edited = self._loader(self.ctf_data[index])
            img = [img, img_edited]
            target = torch.Tensor([target, target])
            sensitive = torch.Tensor([sensitive, 0 if sensitive ==1 else 1])
        else:
            img = [img]
        
        if self.transform is not None:
            img = self.transform(img)
            img = torch.stack(img) if (self.test_pair or self.split == 'train') else img[0]

        return img, 0, sensitive, target, (index, img_name)