import sys
from torch import nn
import torchvision.models.vgg as models2
sys.path.append('../')
from models.networks.network import Net
import torchvision.models.inception as models

import math

class Inceptionv3(Net):

    def __init__(self,cf, num_classes=21,pretrained=False, net_name='Inception'):
        super(Inceptionv3, self).__init__(cf)

        self.url = None
        self.pretrained = pretrained
        self.net_name = net_name
        self.aux_logits=True

        """if pretrained:
    self.model = models.inception_v3(pretrained=pretrained)
    if self.aux_logits:
        self.model.AuxLogits.fc =nn.Linear(768, num_classes)
    self.model.fc = nn.Linear(2048, num_classes)
else:"""
        self.model = models.inception_v3(pretrained=False, num_classes=num_classes)

    def forward(self, x):
        return self.model.forward(x)

    def load_basic_weights(self):
        pass






