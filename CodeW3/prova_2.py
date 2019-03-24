from PIL import Image
import numpy as np
import os.path
import matplotlib
from skimage import img_as_float
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from skimage.color import gray2rgb, rgb2gray


def camvid_colormap():
    colormap = np.zeros((20, 3), dtype=np.uint8)
    colormap[0] = [128, 64, 128]
    colormap[1] = [244, 35, 232]
    colormap[2] = [70, 70, 70]
    colormap[3] = [102, 102, 156]
    colormap[4] = [190, 153, 153]
    colormap[5] = [153, 153, 153]
    colormap[6] = [250, 170, 30]
    colormap[7] = [220, 220,  0]
    colormap[8] = [107, 142, 35]
    colormap[9] = [152, 251, 152]
    colormap[10] = [70, 130, 180]
    colormap[11] = [220, 20, 60]
    colormap[12] = [255, 0,  0]
    colormap[13] = [0, 0, 142]
    colormap[14] = [0, 0, 70]
    colormap[15] = [0, 60, 100]
    colormap[16] = [0, 80, 100]
    colormap[17] = [0, 0, 230]
    colormap[18] = [119, 11, 32]
    colormap[19] = [0, 0, 0]
    return colormap / 256


def my_label2rgb(labels, colors, bglabel=None, bg_color=(0., 0., 0.)):
    output = np.zeros(labels.shape + (3,), dtype=np.float64)
    for i in range(len(colors)):
        if i != bglabel:
            output[(labels == i).nonzero()] = colors[i]
    if bglabel is not None:
        output[(labels == bglabel).nonzero()] = bg_color
    return output


def my_label2rgboverlay(labels, colors, image, bglabel=None,
                        bg_color=(0., 0., 0.), alpha=0.2):
    image_float = gray2rgb(img_as_float(rgb2gray(image)))
    label_image = my_label2rgb(labels, colors, bglabel=bglabel,
                               bg_color=bg_color)
    output = image_float * alpha + label_image * (1 - alpha)
    return output


im1 = Image.open('/home/grupo03/M5/Code/test_fastnet/testFastNetMaxZUZA2/predictions/Seq05VD_f05100.png')
im2 = Image.open('/home/grupo03/M5/Code/test_fastnet/testFastNetMaxZUZA2/Seq05VD_f05100.png')

im1arr = np.asarray(im1)


colors = camvid_colormap()

output1 = my_label2rgb(im1arr, colors)

im2arr = np.asarray(im2)

output2 = my_label2rgboverlay(im1arr, colors, im2arr)

plt.imshow(output1)
plt.savefig('/home/grupo03/M5/Code/test_fastnet/testFastNetMaxZUZA2/city1.png')

plt.imshow(output2)
plt.savefig('/home/grupo03/M5/Code/test_fastnet/testFastNetMaxZUZA2/city2.png')
