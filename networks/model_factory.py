import torch.nn as nn

from networks.resnet import resnet18, resnet152
from networks.shufflenet import shufflenet_v2_x1_0
from networks.cifar_net import Net
from networks.cifar_net_v2 import Net as Net_v2
from networks.mlp import MLP
from networks.resnet_cifar import resnet56


class ModelFactory():
    def __init__(self):
        pass

    @staticmethod
    def get_model(target_model, num_classes, img_size, pretrained=False):

        if target_model == 'resnet56':
            return resnet56(num_classes=num_classes)

        elif target_model == 'resnet':
            if pretrained:
                model = resnet18(pretrained=True)
                model.fc = nn.Linear(in_features=512, out_features=num_classes, bias=True)
            else:
                model = resnet18(pretrained=False, num_classes=num_classes)
            return model

        else:
            raise NotImplementedError

