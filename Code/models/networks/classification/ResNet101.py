import sys
from torch import nn
import torchvision.models.vgg as models2
sys.path.append('../')
from models.networks.network import Net
import torchvision.models.resnet as models

import math

class ResNet101(Net):

    def __init__(self,cf, num_classes=21,zero_init_residual=True, pretrained=False, net_name='ResNet101'):
        super(ResNet101, self).__init__(cf)

        self.url = None
        self.pretrained = pretrained
        self.net_name = net_name

        if pretrained:
            self.model = models.resnet152(pretrained=True)
            self.model.fc = nn.Linear(2048, num_classes)
        else:
            self.model = models.resnet152(pretrained=False, num_classes=num_classes)

    def forward(self, x):
        return self.model.forward(x)

    def load_basic_weights(self):
        pass






