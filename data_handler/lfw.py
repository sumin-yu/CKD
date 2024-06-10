import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from PIL import Image

from torchvision.datasets.utils import check_integrity, download_and_extract_archive, download_url, verify_str_arg
from data_handler.dataset_factory import GenericDataset

import pandas as pd
from functools import partial
from os.path import join

import torch

class _LFW(GenericDataset):

    base_folder = "lfw-py"
    download_url_prefix = "http://vis-www.cs.umass.edu/lfw/"

    file_dict = {
        "original": ("lfw", "lfw.tgz", "a17d05bd522c52d84eca14327a23d494"),
        "funneled": ("lfw_funneled", "lfw-funneled.tgz", "1b42dfed7d15c9b2dd63d5e5840c86ad"),
        "deepfunneled": ("lfw-deepfunneled", "lfw-deepfunneled.tgz", "68331da3eb755a505a502b5aacb3c201"),
    }
    checksums = {
        "pairs.txt": "9f1ba174e4e1c508ff7cdf10ac338a7d",
        "pairsDevTest.txt": "5132f7440eb68cf58910c8a45a2ac10b",
        "pairsDevTrain.txt": "4f27cbf15b2da4a85c1907eb4181ad21",
        "people.txt": "450f0863dd89e85e73936a6d71a3474b",
        "peopleDevTest.txt": "e4bf5be0a43b5dcd9dc5ccfcb8fb19c5",
        "peopleDevTrain.txt": "54eaac34beb6d042ed3a7d883e247a21",
        "lfw-names.txt": "a6d0a479bd074669f656265a6e693f6d",
    }
    annot_file = {"10fold": "", "train": "DevTrain_new", "test": "DevTest", "valid": "DevValid_new"}
    names = "lfw-names.txt"

    def __init__(
        self,
        root: str,
        split: str,
        image_set: str,
        view: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:
        super().__init__(root, transform=transform)

        self.image_set = verify_str_arg(image_set.lower(), "image_set", self.file_dict.keys())
        images_dir, self.filename, self.md5 = self.file_dict[self.image_set]

        self.view = verify_str_arg(view.lower(), "view", ["people", "pairs"])
        self.split = verify_str_arg(split.lower(), "split", ["10fold", "train", "test", "valid"])
        self.labels_file = f"{self.view}{self.annot_file[self.split]}.txt"
        self.data: List[Any] = []

        if download:
            self.download()

        # if not self._check_integrity():
        #     raise RuntimeError("Dataset not found or corrupted. You can use download=True to download it")

        self.images_dir = join(self.root, self.base_folder, images_dir)

    def _loader(self, path: str) -> Image.Image:
        with open(path, "rb") as f:
            img = Image.open(f)
            return img.convert("RGB")

    def _check_integrity(self) -> bool:
        st1 = check_integrity(join(self.root, self.base_folder, self.filename), self.md5)
        st2 = check_integrity(join(self.root, self.base_folder, self.labels_file), self.checksums[self.labels_file])
        if not st1 or not st2:
            return False
        if self.view == "people":
            return check_integrity(join(self.root, self.base_folder, self.names), self.checksums[self.names])
        return True

    def download(self) -> None:
        if self._check_integrity():
            print("Files already downloaded and verified")
            return
        # url = f"{self.download_url_prefix}{self.filename}"
        # download_and_extract_archive(url, self.root, self.base_folder, filename=self.filename, md5=self.md5)
        download_url(f"{self.download_url_prefix}{self.labels_file}", join(self.root, self.base_folder))
        if self.view == "people":
            download_url(f"{self.download_url_prefix}{self.names}", join(self.root, self.base_folder))

    def _get_path(self, identity: str, no: Union[int, str]) -> str:
        return join(self.images_dir, identity, f"{identity}_{int(no):04d}.jpg")
    
    def _get_ctf_path(self, identity: str, no: Union[int, str]) -> str:
        return join(self.images_ctf_dir, identity, f"{identity}_{int(no):04d}.jpg")

    def extra_repr(self) -> str:
        return f"Alignment: {self.image_set}\nSplit: {self.split}"

    def __len__(self) -> int:
        return len(self.data)


class LFWPeople(_LFW):
    """`LFW <http://vis-www.cs.umass.edu/lfw/>`_ Dataset.

    Args:
        root (string): Root directory of dataset where directory
            ``lfw-py`` exists or will be saved to if download is set to True.
        split (string, optional): The image split to use. Can be one of ``train``, ``test``,
            ``10fold`` (default).
        image_set (str, optional): Type of image funneling to use, ``original``, ``funneled`` or
            ``deepfunneled``. Defaults to ``funneled``.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomRotation``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    """

    def __init__(
        self,
        root: str,
        split: str = "10fold",
        image_set: str = "funneled",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False, 
        target_attr: str = "Smiling",
        sen_attr: str = "Male",
        test_set: str = 'original',

    ) -> None:
        super().__init__(root, split, image_set, "people", transform, target_transform, download)


        ######
        fn = partial(join, self.root, self.base_folder)
        if test_set == 'original':
            attr = pd.read_csv(fn("lfw_attributes_binary.txt"), sep='\t') 
            # attr = attr[attr["imagenum"] == 1]
        elif test_set == 'strong_f':
            attr = pd.read_csv(fn("lfw_attributes_binary_test_strong_filter.txt"), sep='\t')
            attr = attr[attr["imagenum"] == 1]
        self.filename = pd.read_csv(fn(self.labels_file), sep='\t').index.tolist()
        self.filename = [f.replace('_', ' ') for f in self.filename]
        attr = attr[attr["person"].isin(self.filename)]
        self.filename = attr["person"].tolist()
        self.filename = [f.replace(' ', '_') for f in self.filename]
        self.filename_num = attr["imagenum"].tolist()
        attr = attr.drop(columns=['person'])
        self.attr = torch.as_tensor(attr.values)

        self.data = self._get_people()

        self.num_classes = 2
        self.num_groups = 2
        print('num classes is {}'.format(self.num_classes))

        self.attr_names = list(attr.columns)
        target_attr = target_attr.replace('_', ' ')
        self.target_idx = self.attr_names.index(target_attr)
        sen_attr = sen_attr.replace('_', ' ')
        self.sensi_idx = self.attr_names.index(sen_attr)
        self.feature_idx = [i for i in range(len(self.attr_names)) if i != self.target_idx and i!=self.sensi_idx]
        self.features = [[int(s), int(l), filename] for s, l, filename in \
                            zip(self.attr[:, self.sensi_idx], self.attr[:, self.target_idx], self.filename)]
        self.n_data, self.idxs_per_group = self._data_count(self.features, self.num_groups, self.num_classes)
        print('num data is {}'.format(len(self.features)))
        print('num data is {}'.format(len(self.data)))
        
        #######
        self.test_pair = False
        self.images_ctf_dir = join(self.root, self.base_folder, "lfw_funneled_Male")
        self.ctf_data = self._get_people_ctf()
    
    def __len__(self) -> int:
        return len(self.features)

    def _get_people(self) -> Tuple[List[str], List[int]]:
        data = []
        # with open(join(self.root, self.base_folder, self.labels_file)) as f:
            # lines = f.readlines()
            # n_folds, s = (int(lines[0]), 1) if self.split == "10fold" else (1, 0)

            # for fold in range(n_folds):
                # n_lines = int(lines[s])
                # people = [line.strip().split("\t") for line in lines[s + 1 : s + n_lines + 1]]
                # s += n_lines + 1
                # for i, (identity, num_imgs) in enumerate(people):
                    # for num in range(1, int(num_imgs) + 1):
        for identity, num in zip(self.filename, self.filename_num):
            # num = 1
            img = self._get_path(identity, num)
            data.append(img)

        return data
    
    def _get_people_ctf(self) -> Tuple[List[str], List[int]]:
        data = []
        for identity, num in zip(self.filename, self.filename_num):
            # num = 1
            img = self._get_ctf_path(identity, num)
            data.append(img)

        return data

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: Tuple (image, target) where target is the identity of the person.
        """
        sensitive, target, img_name = self.features[index]
        img = self._loader(self.data[index])
        if self.test_pair:
            img_edited = self._loader(self.ctf_data[index])
            img = [img, img_edited]
            target = torch.Tensor([target, target])
            sensitive = torch.Tensor([sensitive, 0 if sensitive ==1 else 1])
        else:
            img = [img]
        
        if self.transform is not None:
            img = self.transform(img)
            img = torch.stack(img) if (self.test_pair) else img[0]

        return img, 0, sensitive, target, (index, img_name)

    def extra_repr(self) -> str:
        return super().extra_repr() + f"\nClasses (identities): {len(self.class_to_idx)}"
