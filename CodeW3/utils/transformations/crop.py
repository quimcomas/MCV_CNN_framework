import numpy as np
import random
import torch
import sys
import math
sys.path.append('../')
from utils.box import box_iou, box_clamp

class CropSegSem(object):
    def __init__(self, cf):
        self.cf = cf

    def __call__(self, img, mask):
        if self.cf.crop_train is not None:
            h, w = np.shape(img)[0:2]
            th, tw = self.cf.crop_train
            if w == tw and h == th:
                return img, mask
            elif tw>w and th>h:
                diff_w = tw - w
                marg_w_init = int(diff_w / 2)
                marg_w_fin = diff_w - marg_w_init

                diff_h = th - h
                marg_h_init = int(diff_h / 2)
                marg_h_fin = diff_h - marg_h_init

                tmp_img = np.zeros((th, tw, 3))
                tmp_img[marg_h_init:th - marg_h_fin, marg_w_init:tw - marg_w_fin] = img[0:h, 0:w]
                img = tmp_img

                if type(mask) != type(None):
                  tmp_mask = self.cf.void_class * np.ones((th, tw))
                  tmp_mask[marg_h_init:th - marg_h_fin, marg_w_init:tw - marg_w_fin] = mask[0:h, 0:w]
                  mask = tmp_mask

            elif tw>w:
                diff_w = tw-w
                marg_w_init = int(diff_w/2)
                marg_w_fin = diff_w - marg_w_init
                tmp_img = np.zeros((th, tw ,3))

                y1 = random.randint(0, h - th)
                tmp_img[:,marg_w_init:tw - marg_w_fin] = img[y1:y1 + th,0:w]
                img = tmp_img

                if type(mask) != type(None):
                  tmp_mask = self.cf.void_class*np.ones((th, tw))
                  tmp_mask[:,marg_w_init:tw - marg_w_fin] = mask[y1:y1 + th, 0:w]
                  mask = tmp_mask

            elif th>h:
                diff_h = th - h
                marg_h_init = int(diff_h / 2)
                marg_h_fin = diff_h - marg_h_init
                tmp_img = np.zeros((th, tw, 3))

                x1 = random.randint(0, w - tw)
                tmp_img[marg_h_init:th-marg_h_fin, :] = img[0:h, x1:x1 + tw]
                img = tmp_img

                if type(mask) != type(None):
                  tmp_mask = self.cf.void_class * np.ones((th, tw))
                  tmp_mask[marg_h_init:th-marg_h_fin, :] = mask[0:h, x1:x1 + tw]
                  mask = tmp_mask
            else:
                x1 = random.randint(0, w - tw)
                y1 = random.randint(0, h - th)
                img = img[y1:y1 + th, x1:x1 + tw]
                if type(mask) != type(None):
                  mask = mask[y1:y1 + th, x1:x1 + tw]

        return img, mask

class CropObjDet(object):
    def __init__(self, cf):
        self.cf = cf

    def __call__(self, img, boxes, labels, min_scale=0.3, max_aspect_ratio=2.):
        '''Randomly crop a PIL image.

        Args:
          img: (PIL.Image) image.
          boxes: (tensor) bounding boxes, sized [#obj, 4].
          labels: (tensor) bounding box labels, sized [#obj,].
          min_scale: (float) minimal image width/height scale.
          max_aspect_ratio: (float) maximum width/height aspect ratio.

        Returns:
          img: (PIL.Image) cropped image.
          boxes: (tensor) object boxes.
          labels: (tensor) object labels.
        '''
        imw, imh = img.size
        params = [(0, 0, imw, imh)]  # crop roi (x,y,w,h) out
        for min_iou in (0, 0.1, 0.3, 0.5, 0.7, 0.9):
            for _ in range(100):
                scale = random.uniform(min_scale, 1)
                aspect_ratio = random.uniform(
                    max(1 / max_aspect_ratio, scale * scale),
                    min(max_aspect_ratio, 1 / (scale * scale)))
                w = int(imw * scale * math.sqrt(aspect_ratio))
                h = int(imh * scale / math.sqrt(aspect_ratio))

                x = random.randrange(imw - w)
                y = random.randrange(imh - h)

                roi = torch.tensor([[x, y, x + w, y + h]], dtype=torch.float)
                ious = box_iou(boxes, roi)
                if ious.min() >= min_iou:
                    params.append((x, y, w, h))
                    break

        x, y, w, h = random.choice(params)
        img = img.crop((x, y, x + w, y + h))

        center = (boxes[:, :2] + boxes[:, 2:]) / 2
        mask = (center[:, 0] >= x) & (center[:, 0] <= x + w) \
               & (center[:, 1] >= y) & (center[:, 1] <= y + h)
        if mask.any():
            boxes = boxes[mask] - torch.tensor([x, y, x, y], dtype=torch.float)
            boxes = box_clamp(boxes, 0, 0, w, h)
            labels = labels[mask]
        else:
            boxes = torch.tensor([[0, 0, 0, 0]], dtype=torch.float)
            labels = torch.tensor([0], dtype=torch.long)
        return img, boxes, labels
