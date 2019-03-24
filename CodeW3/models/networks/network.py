import os
import sys
import numpy as np
import wget
import torch
import math
from torch import nn
sys.path.append('../')

class Net(nn.Module):
    def __init__(self, cf):
        super(Net, self).__init__()
        self.url = None
        self.net_name = None
        self.loss = None
        self.optimizer = None
        self.scheduler = None
        self.cf = cf

    def forward(self, x):
        pass

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_type = 'uniform'
                if (init_type == 'he'):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        m.bias.data.uniform_(-1, 1)
                if (init_type == 'xavier'):
                    nn.init.xavier_normal_(m.weight)
                    if m.bias is not None:
                        m.bias.data.uniform_(-1, 1)
                elif(init_type == 'uniform'):
                    n = m.in_channels
                    for k in m.kernel_size:
                        n *= k
                    stdv = 1. / math.sqrt(n)
                    m.weight.data.uniform_(-stdv, stdv)
                    if m.bias is not None:
                        m.bias.data.uniform_(-stdv, stdv)
            if isinstance(m, nn.ConvTranspose2d):
                assert m.kernel_size[0] == m.kernel_size[1]
                initial_weight = self.get_upsampling_weight(
                    m.in_channels, m.out_channels, m.kernel_size[0])
                m.weight.data.copy_(initial_weight)

    # https://github.com/shelhamer/fcn.berkeleyvision.org/blob/master/surgery.py
    def get_upsampling_weight(self, in_channels, out_channels, kernel_size):
        """Make a 2D bilinear kernel suitable for upsampling"""
        factor = (kernel_size + 1) // 2
        if kernel_size % 2 == 1:
            center = factor - 1
        else:
            center = factor - 0.5
        og = np.ogrid[:kernel_size, :kernel_size]
        filt = (1 - abs(og[0] - center) / factor) * \
               (1 - abs(og[1] - center) / factor)
        weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size),
                          dtype=np.float64)
        weight[range(in_channels), range(out_channels), :, :] = filt
        return torch.from_numpy(weight).float()

    def load_basic_weights(self):
        print("load basic weights")
        if not os.path.exists(self.cf.basic_models_path):
            os.makedirs(self.cf.basic_models_path)
        filename = os.path.join(self.cf.basic_models_path, 'basic_'+ self.net_name.lower() +'.pth')
        self.download_if_not_exist(filename)
        self.restore_weights(filename)

    def download_if_not_exist(self, filename):
        # Download the file if it does not exist
        # print(self.url)
        # print(not os.path.isfile(filename))
        if not os.path.isfile(filename) and self.url is not None:
            print("downloading file")
            wget.download(self.url, filename)
        else:
            print("NOT downloading file")

    def restore_weights(self, filename):
        print('\t Restoring weight from ' + filename)
        if self.cf.model_type.lower() == 'ssd512':
            self.load_state_dict(torch.load(filename), strict=False)
        else:
            self.load_state_dict2(torch.load(filename))

    def load_state_dict2(self, pretrained_dict):
        model_dict = self.state_dict()

        for k, v in pretrained_dict.items():
            if v.size() != model_dict[k].size():
                print('\t WARNING: Could not load layer ' + str(k) + ' with shape: ' + str(v.size()) + ' and ' + str(
                    model_dict[k].size()))
            else:
                model_dict[k] = v

        super(Net, self).load_state_dict(model_dict)

    def restore_weights2(self, filename):
        print('\t Restoring weight from ' + filename)
        self.load_state_dict2(torch.load(os.path.join(filename))['model_state_dict'])