import torch
import os
import numpy as np
import wget
import sys
import json
import copy
from torch import nn
sys.path.append('../')
from utils.statistics import Statistics

class Model(nn.Module):
    def __init__(self, cf):
        super(Model, self).__init__()
        self.url = None
        self.net_name = None
        self.cf = cf
        self.best_stats = Statistics()

    def forward(self, x):
        pass

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.zero_()
                if m.bias is not None:
                    m.bias.data.zero_()
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
        if not os.path.exists(self.cf.basic_models_path):
            os.makedirs(self.cf.basic_models_path)
        filename = os.path.join(self.cf.basic_models_path, 'basic_'+ self.net_name.lower() +'.pth')
        self.download_if_not_exist(filename)
        self.restore_weights(filename)

    def download_if_not_exist(self, filename):
        # Download the file if it does not exist
        if not os.path.isfile(filename) and self.url is not None:
            wget.download(self.url, filename)

    def restore_weights(self, filename):
        print('\t Restoring weight from ' + filename)
        self.load_state_dict(torch.load(os.path.join(filename)))

    def load_state_dict(self, pretrained_dict):
        model_dict = self.state_dict()

        for k, v in pretrained_dict.items():
            if v.size() != model_dict[k].size():
                print('\t WARNING: Could not load layer ' + str(k) + ' with shape: ' + str(v.size()) + ' and ' + str(
                    model_dict[k].size()))
            else:
                model_dict[k] = v

        super(Model, self).load_state_dict(model_dict)

    def restore_weights2(self, filename):
        print('\t Restoring weight from ' + filename)
        self.load_state_dict(torch.load(os.path.join(filename))['model_state_dict'])

    def save_model(self):
        if self.cf.save_weight_only:
            torch.save(self.state_dict(), os.path.join(self.cf.output_model_path,
                self.cf.model_name + '.pth'))
        else:
            torch.save(self, os.path.join(self.cf.exp_folder, self.cf.model_name + '.pth'))

    def save(self, stats):
        if self.cf.save_condition == 'always':
            save = True
        else:
            save = self.check_stat(stats)
        if save:
            self.save_model()
            self.best_stats = copy.deepcopy(stats)
        return save

    def check_stat(self, stats):
        check = False
        if self.cf.save_condition.lower() == 'train_loss':
            if stats.train.loss < self.best_stats.train.loss:
                check = True
        elif self.cf.save_condition.lower() == 'valid_loss':
            if stats.val.loss < self.best_stats.val.loss:
                check = True
        elif self.cf.save_condition.lower() == 'valid_miou':
            if stats.val.mIoU > self.best_stats.val.mIoU:
                check = True
        elif self.cf.save_condition.lower() == 'valid_macc':
            if stats.val.acc > self.best_stats.val.acc:
                check = True
        elif self.cf.save_condition.lower() == 'precision':
            if stats.val.precision > self.best_stats.val.precision:
                check = True
        elif self.cf.save_condition.lower() == 'recall':
            if stats.val.recall > self.best_stats.val.recall:
                check = True
        elif self.cf.save_condition.lower() == 'f1score':
            if stats.val.f1score > self.best_stats.val.f1score:
                check = True
        return check

    def restore_model(self):
        print('\t Restoring weight from ' + self.cf.input_model_path + self.cf.model_name)
        net = torch.load(os.path.join(self.cf.input_model_path, self.cf.model_name + '.pth'))
        return net

    def load_statistics(self):
        if os.path.exists(self.cf.best_json_file):
            with open(self.cf.best_json_file) as json_file:
                json_data = json.load(json_file)
                self.best_stats.epoch = json_data[0]['epoch']
                self.best_stats.train = self.fill_statistics(json_data[0],self.best_stats.train)
                self.best_stats.val = self.fill_statistics(json_data[1], self.best_stats.val)

    def fill_statistics(self, dict_stats, stats):
        stats.loss = dict_stats['loss']
        stats.mIoU = dict_stats['mIoU']
        stats.acc = dict_stats['acc']
        stats.precision = dict_stats['precision']
        stats.recall = dict_stats['recall']
        stats.f1score = dict_stats['f1score']
        stats.conf_m = dict_stats['conf_m']
        stats.mIoU_perclass = dict_stats['mIoU_perclass']
        stats.acc_perclass = dict_stats['acc_perclass']
        stats.precision_perclass = dict_stats['precision_perclass']
        stats.recall_perclass = dict_stats['recall_perclass']
        stats.f1score_perclass = dict_stats['f1score_perclass']
        return stats