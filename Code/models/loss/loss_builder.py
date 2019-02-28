from segmentation.crossEntropyLoss2d import CrossEntropyLoss2d
from classification.crossEntropyLoss import CrossEntropyLoss
from detection.ssd_loss import SSDLoss

class Loss_Builder():
    def __init__(self, cf):
        self.cf = cf
        self.loss_manager = None

    def build(self):
        if self.cf.loss_type.lower() == 'cross_entropy_segmentation':
            self.loss_manager = CrossEntropyLoss2d(self.cf, ignore_index=self.cf.void_class)
        elif self.cf.loss_type.lower() == 'cross_entropy_classification':
            self.loss_manager = CrossEntropyLoss(self.cf, ignore_index=self.cf.void_class)
        elif self.cf.loss_type.lower() == 'ssd_loss':
            self.loss_manager = SSDLoss(self.cf)
        else:
            raise ValueError('Unknown loss type')
        return self.loss_manager