import numpy as np
import random
from PIL import Image

class RandomHorizontalFlipSegSem(object):
    def __init__(self, cf):
        self.cf = cf

    def __call__(self, img, gt):
        if self.cf.hflips and random.random() < 0.5:
            return np.fliplr(img), np.fliplr(gt)
        return img, gt

class RandomHorizontalFlipObjDet(object):
    def __init__(self, cf):
        self.cf = cf

    def __call__(self, img, boxes, labels):
        '''Randomly flip PIL image.

        If boxes is not None, flip boxes accordingly.

        Args:
          img: (PIL.Image) image to be flipped.
          boxes: (tensor) object boxes, sized [#obj,4].

        Returns:
          img: (PIL.Image) randomly flipped image.
          boxes: (tensor) randomly flipped boxes.
        '''
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            w = img.width
            if boxes is not None:
                xmin = w - boxes[:,2]
                xmax = w - boxes[:,0]
                boxes[:,0] = xmin
                boxes[:,2] = xmax
        return img, boxes, labels