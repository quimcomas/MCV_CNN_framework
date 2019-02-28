import numpy as np
import PIL

class rescale(object):
    def __init__(self, rescale):
        self.rescale = rescale

    def __call__(self, image):
        return image * self.rescale


class mean_norm(object):
    def __init__(self, mean):
        self.mean = np.asarray(mean, dtype=np.float32)

    def __call__(self, image):
        return image - self.mean


class std_norm(object):
    def __init__(self, std):
        self.std = np.asarray(std, dtype=np.float32)

    def __call__(self, image):
        return image / (self.std + 1e-7)


class preproces_input(object):
    def __init__(self, cf):
        self.cf = cf
        if self.cf.rescale is not None:
            self.rescale = rescale(self.cf.rescale)
        if self.cf.mean is not None:
            self.mean = mean_norm(self.cf.mean)
        if self.cf.std is not None:
            self.std = std_norm(self.cf.std)

    def __call__(self, image):
        if type(image).__name__ == 'Image':
            pil = True
            image = np.array(image)
        else:
            pil = False
        if self.cf.rescale is not None:
            image = self.rescale(image)
        if self.cf.mean is not None:
            image = self.mean(image)
        if self.cf.std is not None:
            image = self.std(image)
        if pil:
            image = PIL.Image.fromarray(np.uint8(image))
        return image