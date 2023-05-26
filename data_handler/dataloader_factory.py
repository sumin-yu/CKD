from data_handler.dataset_factory import DatasetFactory

import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader


class DataloaderFactory:
    def __init__(self):
        pass

    @staticmethod
    def get_dataloader(name, img_size=224, batch_size=256, seed = 0, num_workers=4,
                       target='Smiling', skew_ratio=1., labelwise=False, num_aug=1, tuning=False):

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
        
        if name == 'celeba':
            transform_list = [transforms.RandomResizedCrop(img_size),
                              transforms.RandomHorizontalFlip(),
                              transforms.ToTensor(),
                              normalize
                             ]
        elif name == 'cifar10_indiv':
            transform_list = [transforms.ToPILImage(),
                            #   transforms.Resize((38,38)),
                            #   transforms.RandomApply([transforms.RandomRotation(30),transforms.CenterCrop(32)], p=1.0),
                            #   transforms.RandomHorizontalFlip(),
                            #   transforms.Resize((32,32)),
                              transforms.ToTensor()]
        elif name == 'cifar10':
            transform_list = [transforms.ToPILImage(),
                            #   transforms.RandomHorizontalFlip(),
                              transforms.ToTensor()
                              ]
        else:
            transform_list = [transforms.Resize((256,256)),
                              transforms.RandomCrop(img_size),
                              transforms.RandomHorizontalFlip(),
                              transforms.ToTensor(),
                              normalize
                             ]

        if 'cifar10' in name:
            test_transform_list = [transforms.ToTensor()]
        else:
            test_transform_list = [transforms.Resize((img_size,img_size)),
                                  transforms.ToTensor(),
                                  normalize]
        preprocessing = transforms.Compose(transform_list)
        test_preprocessing = transforms.Compose(test_transform_list)

        if tuning:
            train_dataset = DatasetFactory.get_dataset(name, transform=preprocessing, split='train', target=target,
                                                       seed=seed, skew_ratio=skew_ratio, labelwise=labelwise, num_aug=num_aug, tuning=tuning)
            val_dataset = DatasetFactory.get_dataset(name, transform=test_preprocessing, split='val', target=target,
                                                        seed=seed, skew_ratio=skew_ratio, labelwise=labelwise, num_aug=num_aug, tuning=tuning)
        else:
            train_dataset = DatasetFactory.get_dataset(name, transform=preprocessing, split='train', target=target,
                                                       seed=seed, skew_ratio=skew_ratio, labelwise=labelwise, num_aug=num_aug)
        
        test_dataset = DatasetFactory.get_dataset(name, transform=test_preprocessing, split='test', target=target,
                                                  seed=seed, skew_ratio=skew_ratio)

        def _init_fn(worker_id):
            np.random.seed(int(seed))
            
        num_classes = test_dataset.num_classes
        num_groups = test_dataset.num_groups

        if labelwise:
            from data_handler.custom_loader import Customsampler
            sampler = Customsampler(train_dataset, replacement=False, batch_size=batch_size)
            train_dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler,
                                          num_workers=num_workers, worker_init_fn=_init_fn, pin_memory=True, drop_last=True)
        else:
            train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                          num_workers=num_workers, worker_init_fn=_init_fn, pin_memory=True, drop_last=True)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                     num_workers=num_workers, worker_init_fn=_init_fn, pin_memory=True)
        
        if tuning:
            val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                                        num_workers=num_workers, worker_init_fn=_init_fn, pin_memory=True)
            return num_classes, num_groups, train_dataloader, val_dataloader, test_dataloader
        else:
            return num_classes, num_groups, train_dataloader, test_dataloader
        # print('Dataset loaded.')

