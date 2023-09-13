from os.path import join
import pandas
import numpy as np
from functools import partial
from torchvision.datasets.utils import download_file_from_google_drive, check_integrity, verify_str_arg
from data_handler.celeba import CelebA

class CelebA_pseudo(CelebA):
    def __init__(self, **kwargs):
        super(CelebA_pseudo, self).__init__(**kwargs)
        if self.split == 'train':
            split_map = {
            "train": 0,
            "valid": 1,
            "test": 2,
            "all": None,
            }
            split = split_map[verify_str_arg(self.split.lower(), "split",
                                    ("train", "valid", "test", "all" ))]
            fn = partial(join, self.root, self.base_folder)
            attr = pandas.read_csv(fn("list_pseudo_male_celeba.csv"), header=0)
            sensitive = attr.values[:, 2]
            self.features = [[s, i[1], i[2]] for i, s in zip(self.features, sensitive)]
            self.n_data, self.idxs_per_group = self._data_count(self.features, self.num_groups, self.num_classes)
