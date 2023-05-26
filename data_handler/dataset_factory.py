class DatasetFactory:
    def __init__(self):
        pass

    @staticmethod
    def get_dataset(name, transform=None, split='train', target='Attractive', seed=0, skew_ratio=1., labelwise=False, num_aug=1, tuning=False):

        if name == "utkface":
            from data_handler.utkface import UTKFaceDataset
            root = './data/UTKFace'
            return UTKFaceDataset(root=root, split=split, transform=transform,
                                  labelwise=labelwise)

        elif name == "celeba":
            from data_handler.celeba import CelebA
            root='./data/'
            return CelebA(root=root, split=split, transform=transform, target_attr=target, labelwise=labelwise)
        
        elif name == "cifar10":
            from data_handler.cifar10 import CIFAR_10S
            root = './data/cifar10'
            return CIFAR_10S(root=root, split=split, transform=transform, seed=seed, skewed_ratio=skew_ratio,
                             labelwise=labelwise, tuning=tuning)
        elif name == "cifar10_indiv":
            from data_handler.cifar10_indiv import CIFAR_10S
            root = './data/cafar10'
            return CIFAR_10S(root=root, split=split, transform=transform, seed=seed, skewed_ratio=skew_ratio,
                             labelwise=labelwise, num_aug=num_aug, tuning=tuning)
