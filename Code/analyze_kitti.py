import os
import numpy as np
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt

dataset_name = 'KITTI'

classes = ('Car', 'Cyclist', 'Pedestrian', 'Person_sitting', 'Tram', 'Truck', 'Van', 'background')

path_train = '/home/mcv/datasets/M5/classification/' + dataset_name + '/train/'
path_valid = '/home/mcv/datasets/M5/classification/' + dataset_name + '/valid/'
# path_test = '/home/mcv/datasets/M5/classification/' + dataset_name + '/test/'

train_vec = np.array([])
valid_vec = np.array([])
# test_vec = np.array([])

# for fold in train_fold:
for c in range(0, len(classes)):
    train_images = np.size(os.listdir(os.path.join(path_train, classes[c])))
    valid_images = np.size(os.listdir(os.path.join(path_valid, classes[c])))
    # test_images = np.size(os.listdir(os.path.join(path_test, classes[c])))
    train_vec = np.append(train_vec, train_images)
    valid_vec = np.append(valid_vec, valid_images)
    # test_vec = np.append(test_vec, test_images)

fig = plt.figure(1)
ax = fig.add_subplot(111)
ax.bar(range(0, len(train_vec)), train_vec)
ax.set_xlabel('classes')
ax.set_ylabel('number of images')
fig.savefig('hist_train.png')
plt.close(fig)

fig = plt.figure(2)
ax = fig.add_subplot(111)
ax.bar(range(0, len(valid_vec)), valid_vec)
ax.set_xlabel('classes')
ax.set_ylabel('number of images')
fig.savefig('hist_valid.png')
plt.close(fig)

# fig = plt.figure(3)
# ax = fig.add_subplot(111)
# ax.bar(range(0, len(test_vec)), test_vec)
# ax.set_xlabel('classes')
# ax.set_ylabel('number of images')
# fig.savefig('hist_test.png')
# plt.close(fig)