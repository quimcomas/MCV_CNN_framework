import torch
import numpy as np
import os

from dataloader import Data_loader

class fromFileDatasetDetection(Data_loader):

    def __init__(self, cf, image_txt, gt_txt, num_images, resize=None,
                        preprocess=None, transform=None, valid=False, box_coder=None):
        super(fromFileDatasetDetection, self).__init__()
        self.cf = cf
        self.box_coder = box_coder
        self.resize = resize
        self.transform = transform
        self.preprocess = preprocess
        self.num_images = num_images
        self.boxes = []
        self.labels = []
        self.fnames = []
        print ("\t Images from: " + image_txt)
        with open(image_txt) as f:
            image_names = f.readlines()
        # remove whitespace characters like `\n` at the end of each line
        self.image_names = [x.strip() for x in image_names]
        print ("\t Gt from: " + gt_txt)
        with open(gt_txt) as f:
            gt_names = f.readlines()
        self.gt_names = [x.strip() for x in gt_names]
        if self.cf.dataset.lower() == 'kitti':
            self.KittiGT(self.gt_names)
        elif self.cf.dataset.lower() == 'synthiavz':
            self.SynthiaVZGT(self.gt_names)
        if len(self.boxes) == 0:
            raise ValueError('No bounding boxes found for the classes specified in the Dataset')
        if len(self.gt_names) != len(self.image_names):
            raise ValueError('number of images != number GT images')
        print ("\t Images found: " + str(len(self.image_names)))
        if len(self.image_names) < self.num_images or self.num_images == -1:
            self.num_images = len(self.image_names)
        self.img_indexes = np.arange(len(self.image_names))
        self.update_indexes(valid=valid)

    def __len__(self):
        return self.num_images

    def __getitem__(self, idx):
        img_path = self.image_names[self.indexes[idx]]
        img = self.Load_image(img_path, resize=None, grayscale=self.cf.grayscale)
        boxes = self.boxes[idx].clone()  # used clone to avoid any potential change.
        labels = self.labels[idx].clone()
        if self.transform is not None:
            img, boxes, labels = self.transform(img, boxes, labels)
        if self.preprocess is not None:
            img = self.preprocess(img)
        if self.box_coder is not None:
            boxes, labels = self.box_coder.encode(boxes, labels)
        return img, boxes, labels

    def update_indexes(self, num_images=None, valid=False):
        if self.cf.shuffle and not valid:
            np.random.shuffle(self.img_indexes)
        if num_images is not None:
            if len(self.image_names) < self.num_images or num_images == -1:
                self.num_images = len(self.image_names)
            else:
                self.num_images = num_images
        self.indexes = self.img_indexes[:self.num_images]

    def KittiGT(self, filenames):
        # self.classes_list = ['Car', 'Pedestrian', 'Cyclist']
        self.class_to_ind = dict(zip(self.cf.labels, xrange(len(self.cf.labels))))
        for filename in filenames:
            with open(filename) as f:
                box = []
                label = []
                for x in f.readlines():
                    line_splited = x.split(' ')
                    if line_splited[0] in self.cf.labels:
                        label.append(self.class_to_ind[line_splited[0]])
                        xmin = float(line_splited[4])
                        ymin = float(line_splited[5])
                        xmax = float(line_splited[6])
                        ymax = float(line_splited[7])
                        box.append([xmin,ymin,xmax,ymax])
                if len(box) > 0:
                    self.boxes.append(torch.Tensor(box))
                    self.labels.append(torch.LongTensor(label))
                    self.fnames.append(filename)

    def SynthiaVZGT(self, filenames):
        # self.classes_list = ['Car', 'Pedestrian', 'Truck']
        self.class_to_ind = dict(zip(self.cf.labels, xrange(len(self.cf.labels))))
        self.mapping = {14: 'Car', 12: 'Pedestrian', 15: 'Truck'}
        for filename in filenames:
            with open(filename) as f:
                box = []
                label = []
                for x in f.readlines():
                    line_splited = x.split(',')
                    if int(line_splited[4]) in self.mapping:
                        xmin = float(line_splited[0]) if float(line_splited[0]) > -1 else 0.
                        if xmin > 480:
                            xmin = 480.
                        ymin = float(line_splited[3]) if float(line_splited[3]) > -1 else 0.
                        if ymin > 480:
                            ymin = 480.
                        xmax = float(line_splited[2]) if float(line_splited[2]) < 481 else 480.
                        if xmax < 0:
                            xmax = 0.
                        ymax = float(line_splited[1]) if float(line_splited[1]) < 481 else 480.
                        if ymax < 0:
                            ymax = 0.
                        if (xmax-xmin)*(ymax-ymin) > 0:
                            label.append(self.class_to_ind[self.mapping[int(line_splited[4])]])
                            box.append([xmin,ymin,xmax,ymax])
                if len(box) > 0:
                    self.boxes.append(torch.Tensor(box))
                    self.labels.append(torch.LongTensor(label))
                    self.fnames.append(filename)