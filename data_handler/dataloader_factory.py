from data_handler.dataset_factory import DatasetFactory

import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader
import torchvision.transforms.functional as TF

import random 
from functools import partial

def custom_transform(img_list, 
                        arr2img=False, 
                        resize=False, img_resize=256,
                        rand_crop=False, img_size=(224,224),
                        flip_horizon=False, prob=0.5,
                        normalize=False, mean=None, std=None):
    if arr2img:
        img_list = [TF.to_pil_image(img) for img in img_list]

    if resize:
        resize = transforms.Resize(size=img_size)
        img_list = [resize(img) for img in img_list]

    if rand_crop:
        i, j, h, w = transforms.RandomCrop.get_params(
        img_list[0], output_size=(img_size, img_size))
        img_list = [TF.crop(img, i, j, h, w) for img in img_list]

    if flip_horizon:
        if random.random() > prob:
            img_list = [TF.hflip(img) for img in img_list]

    img_list =  [TF.to_tensor(img) for img in img_list]

    if normalize:
        img_list = [TF.normalize(img, mean=mean, std=std) for img in img_list]
    
    return img_list

class DataloaderFactory:
    def __init__(self):
        pass

    @staticmethod
    def get_dataloader(name, img_size=224, batch_size=256, seed = 0, num_workers=4,
                       target='Blond_Hair', sensitive='Male', skew_ratio=1., labelwise=False, method=None):

        # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    #  std=[0.229, 0.224, 0.225])
        train_transform = None
        valid_transform = None
        test_transform = None
        if 'celeba' in name:
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]

            train_transform = partial(custom_transform, 
                                      rand_crop=True, img_size=224,
                                      flip_horizon=True,
                                      normalize=True, mean=mean, std=std)
            
            test_transform = partial(custom_transform, 
                                     resize=True, 
                                     normalize=True, mean=mean, std=std)

            valid_transform = test_transform
            # transform_list = [transforms.RandomResizedCrop(img_size),
            #                   transforms.RandomHorizontalFlip(),
            #                   transforms.ToTensor(),
            #                   normalize
            #                  ]
            
        elif name == 'cifar10_aug':
            train_transform = transforms.Compose([transforms.ToPILImage(),
                            #   transforms.Resize((38,38)),
                            #   transforms.RandomApply([transforms.RandomRotation(30),transforms.CenterCrop(32)], p=1.0),
                            #   transforms.RandomHorizontalFlip(),
                            #   transforms.Resize((32,32)),
                              transforms.ToTensor()])
            
        elif name == 'cifar10':
            train_transform = transforms.Compose([transforms.ToPILImage(),
                            #   transforms.RandomHorizontalFlip(),
                              transforms.ToTensor()
                              ])
            
        elif 'spucobirds' in name:
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
            train_transform = partial(custom_transform,
                                      resize=True, 
                                      rand_crop=True, img_size=224,
                                      flip_horizon=True,
                                      normalize=True, mean=mean, std=std)
            test_transform = partial(custom_transform,
                                     resize=True, 
                                      normalize=True, mean=mean, std=std)
            valid_transform = test_transform
        
        # else:
        #     transform_list = [transforms.Resize((256,256)),
        #                       transforms.RandomCrop(img_size),
        #                       transforms.RandomHorizontalFlip(),
        #                       transforms.ToTensor(),
        #                       normalize
        #                      ]

        if 'cifar10' in name:
            test_transform = transforms.Compose([transforms.ToTensor()])
            valid_transform = test_transform
        # else:
        #     test_transform_list = [transforms.Resize((img_size,img_size)),
        #                           transforms.ToTensor(),
        #                           normalize]
            
        # preprocessing = transforms.Compose(transform_list)
        # test_preprocessing = transforms.Compose(test_transform_list)

        val_dataset = DatasetFactory.get_dataset(name, transform=valid_transform, split='valid', target=target, sensitive=sensitive,
                                                    seed=seed, skew_ratio=skew_ratio)
        train_dataset = DatasetFactory.get_dataset(name, transform=train_transform, split='train', target=target, sensitive=sensitive,
                                                    seed=seed, skew_ratio=skew_ratio, method=method)
            
        test_dataset = DatasetFactory.get_dataset(name, transform=test_transform, split='test', target=target, sensitive=sensitive,
                                                seed=seed, skew_ratio=skew_ratio)

        def _init_fn(worker_id):
            np.random.seed(int(seed))
            
        n_classes = test_dataset.num_classes
        n_groups = test_dataset.num_groups

        if labelwise:
            from torch.utils.data.sampler import WeightedRandomSampler
            weights = train_dataset.make_weights(dataset=name)
            sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
            train_dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler,
                                          num_workers=num_workers, worker_init_fn=_init_fn, pin_memory=True, drop_last=True)
        else:
            train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                          num_workers=num_workers, worker_init_fn=_init_fn, pin_memory=True, drop_last=True)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                     num_workers=num_workers, worker_init_fn=_init_fn, pin_memory=True)
        
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                                    num_workers=num_workers, worker_init_fn=_init_fn, pin_memory=True)
        return n_classes, n_groups, train_dataloader, val_dataloader, test_dataloader
