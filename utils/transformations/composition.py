
class ComposeSemSeg(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, mask):
        for t in self.transforms:
            img, mask = t(img, mask)
        return img, mask

class ComposeObjDet(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, boxes, labels):
        for t in self.transforms:
            img, boxes, labels = t(img, boxes, labels)
        return img, boxes, labels