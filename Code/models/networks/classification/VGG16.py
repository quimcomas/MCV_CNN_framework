import sys
from torch import nn
import torchvision.models.vgg as models
sys.path.append('../')
from models.networks.network import Net
import math

class VGG16(Net):

    def __init__(self, cf, num_classes=21, pretrained=False, net_name='vgg16'):
        super(VGG16, self).__init__(cf)

        self.url = 'http://datasets.cvc.uab.es/models/pytorch/basic_vgg16.pth'
        self.pretrained = pretrained
        self.net_name = net_name

        self.model = models.vgg16(pretrained=False, num_classes=num_classes)

        self.features = nn.Sequential(
            # Block: conv1
            nn.Conv2d(3, 64, kernel_size=3, padding=1),                 # -30
            nn.ReLU(inplace=True),                                      # -29
            nn.Conv2d(64, 64, kernel_size=3, padding=1),                # -28
            nn.ReLU(inplace=True),                                      # -27
            nn.MaxPool2d(kernel_size=2, stride=2),                      # -26

            # Block: conv2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),               # -25
            nn.ReLU(inplace=True),                                      # -24
            nn.Conv2d(128, 128, kernel_size=3, padding=1),              # -23
            nn.ReLU(inplace=True),                                      # -22
            nn.MaxPool2d(kernel_size=2, stride=2),                      # -21

            # Block: conv3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),              # -20
            nn.ReLU(inplace=True),                                      # -19
            nn.Conv2d(256, 256, kernel_size=3, padding=1),              # -18
            nn.ReLU(inplace=True),                                      # -17
            nn.Conv2d(256, 256, kernel_size=3, padding=1),              # -16
            nn.ReLU(inplace=True),                                      # -15
            nn.MaxPool2d(kernel_size=2, stride=2),                      # -14

            # Block: conv4
            nn.Conv2d(256, 512, kernel_size=3, padding=1),              # -13
            nn.ReLU(inplace=True),                                      # -12
            nn.Conv2d(512, 512, kernel_size=3, padding=1),              # -11
            nn.ReLU(inplace=True),                                      # -10
            nn.Conv2d(512, 512, kernel_size=3, padding=1),              # -9
            nn.ReLU(inplace=True),                                      # -8
            nn.MaxPool2d(kernel_size=2, stride=2),                      # -7

            # Block: conv5
            nn.Conv2d(512, 512, kernel_size=3, padding=1),              # -6
            nn.ReLU(inplace=True),                                      # -5
            nn.Conv2d(512, 512, kernel_size=3, padding=1),              # -4
            nn.ReLU(inplace=True),                                      # -3
            nn.Conv2d(512, 512, kernel_size=3, padding=1),              # -2
            nn.ReLU(inplace=True),                                      # -1
            nn.MaxPool2d(kernel_size=2, stride=2)                       # 0
        )

        self.classifier = nn.Sequential(
            # Block: classifier
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes)
        )

        '''if pretrained:
            self.load_basic_weights(net_name)
        else:
            self._initialize_weights()'''

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
