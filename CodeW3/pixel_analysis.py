import os.path
import numpy as np
import matplotlib

matplotlib.use('Agg')
from matplotlib import pyplot as plt
import cv2


def pixels_class_dataset(path, num_classes):
    pixels_class = np.zeros(num_classes)
    for image in os.listdir(path):
        gt = cv2.imread(os.path.join(path, image), -1)
        hist, bin = np.histogram(gt, np.arange(num_classes + 1) - 0.5)
        pixels_class = pixels_class + hist
        pixels_class.astype(float)
    norm_pixels_class = pixels_class / sum(pixels_class)

    return norm_pixels_class


classes = {0: 'background', 1: 'aeroplane', 2: 'bicycle', 3: 'bird', 4: 'boat', 5: 'bottle', 6: 'bus', 7: 'car',
           8: 'cat', 9: 'chair', 10: 'cow', 11: 'diningtable', 12: 'dog', 13: 'horse', 14: 'motorbike', 15: 'person',
           16: 'pottedplant', 17: 'sheep',
           18: 'sofa', 19: 'train', 20: 'tvmonitor', 21: 'void'}

# classes = {0: 'sky', 1: 'building', 2: 'column_pole', 3: 'road', 4: 'sidewalk',
#           5: 'tree', 6: 'sign', 7: 'fence', 8: 'car', 9: 'pedestrian', 10: 'byciclist',11: 'void'}
"""
classes = {0:'road',1:'sidewalk',2:'building',3:'wall',4:'fence',5:'pole',6:'traffic light',7:'traffic sign',8:'vegetation',9:'terrain',10:'sky',
11:'person',12:'rider',13:'car',14:'truck',15:'bus',16:'train',17:'motorcycle',18:'bicycle',19:'void'}

classes = {0:'road',1:'sidewalk',2:'building',3:'wall',4:'fence',5:'pole',6:'traffic light',7:'traffic, sign',
                                8:'vegetation',9:'terrain',10:'sky',11:'person',12:'rider',13:'car',14:'truck',15:'bus',16:'train',
                                17:'motorcycle',18:'bicycle'}
"""
num_classes = len(classes)

train_path = "/home/mcv/datasets/M5/segmentation/pascal2012/train/masks"

validation_path = "/home/mcv/datasets/M5/segmentation/pascal2012/valid/masks"

#test_path = "/home/mcv/datasets/M5/segmentation/synthia_rand_cityscapes/test/masks"

num_train = len([image for image in os.listdir(train_path) if os.path.isfile(os.path.join(train_path, image))])
print(num_train)

num_valid = len(
    [image for image in os.listdir(validation_path) if os.path.isfile(os.path.join(validation_path, image))])
print(num_valid)

#num_test = len([image for image in os.listdir(test_path) if os.path.isfile(os.path.join(test_path, image))])
#print(num_test)

train = pixels_class_dataset(train_path, num_classes)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.bar(range(0, len(train)), train)
ax.set_xlabel('classes')
ax.set_ylabel('Pixels')
ax.set_title('Pixel distribution on training set')
fig.savefig('histogram_training_pascal.png')
plt.close(fig)

validation = pixels_class_dataset(validation_path, num_classes)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.bar(range(0, len(validation)), validation)
ax.set_xlabel('classes')
ax.set_ylabel('Pixels')
ax.set_title('Pixel distribution on validation set')
fig.savefig('histogram_validation_pascal.png')
plt.close(fig)

"""
test = pixels_class_dataset(test_path, num_classes)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.bar(range(0, len(test)), test)
ax.set_xlabel('classes')
ax.set_ylabel('Pixels')
ax.set_title('Pixel distribution on test set')
fig.savefig('histogram_test_pascal.png')
plt.close(fig)
"""
