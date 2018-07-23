import torch
import os
import sys
import json
import copy

sys.path.append('../')
from models.networks.segmentation.FCN8 import FCN8
from models.networks.segmentation.FCN8AtOnce import FCN8AtOnce
from models.networks.segmentation.FCdenseNetTorch import FCDenseNet
from models.networks.classification.VGG16 import VGG16
from models.networks.detection.rpn import RPN
from models.loss.loss_builder import Loss_Builder
from models.optimizer.optimizer_builder import Optimizer_builder
from models.scheduler.scheduler_builder import scheduler_builder
from utils.statistics import Statistics

class Model_builder():
    def __init__(self, cf):
        self.cf = cf
        self.net = None
        self.loss = None
        self.optimizer = None
        self.scheduler = None
        self.best_stats = Statistics()
        
    def build(self):
        if self.cf.pretrained_model.lower() == 'custom' and not self.cf.load_weight_only:
            self.net = self.restore_model()
            return self.net

        if self.cf.model_type.lower() == 'densenetfcn':
            self.net = FCDenseNet(self.cf, nb_layers_per_block=self.cf.model_layers,
                                growth_rate=self.cf.model_growth,
                                nb_dense_block=self.cf.model_blocks, 
                                n_channel_start=48,
                                n_classes=self.cf.num_classes,
                                drop_rate=0, bottle_neck=False).cuda()
        elif self.cf.model_type.lower() == 'fcn8':
            self.net = FCN8(self.cf, num_classes=self.cf.num_classes, pretrained=self.cf.basic_pretrained_model).cuda()
        elif self.cf.model_type.lower() == 'fcn8atonce':
            self.net = FCN8AtOnce(self.cf, num_classes=self.cf.num_classes, pretrained=self.cf.basic_pretrained_model).cuda()
        elif self.cf.model_type.lower() == 'rpn':
            self.net = RPN(self.cf, 512)
        elif self.cf.model_type.lower() == 'vgg16':
            self.net = VGG16(self.cf, num_classes=self.cf.num_classes, pretrained=self.cf.basic_pretrained_model).cuda()
        else:
            raise ValueError('Unknown model')

        print(self.cf.resume_experiment)
        print((self.cf.pretrained_model.lower() == 'custom' and self.cf.load_weight_only))
        if self.cf.resume_experiment or (self.cf.pretrained_model.lower() == 'custom' and self.cf.load_weight_only):
            self.net.restore_weights(os.path.join(self.cf.input_model_path))
            if self.cf.resume_experiment:
                self.load_statistics()
        elif self.net.pretrained:
            self.net.load_basic_weights()
        else:
            self.net.initialize_weights()

        # Loss definition
        if self.loss is None:
            self.loss = Loss_Builder(self.cf).build().cuda()

        # Optimizer definition
        self.optimizer = Optimizer_builder().build(self.cf, self.net)

        # Learning rate scheduler
        self.scheduler = scheduler_builder().build(self.cf, self.optimizer)

    def save_model(self):
        if self.cf.save_weight_only:
            torch.save(self.net.state_dict(), os.path.join(self.cf.output_model_path,
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
        


                   