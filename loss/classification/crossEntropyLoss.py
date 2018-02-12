import torch.nn.functional as F
import torch.nn as nn
from classification_loss import Classification_Loss


class CrossEntropyLoss(Classification_Loss):
    def __init__(self, cf, weight=None, ignore_index=255):
        super(CrossEntropyLoss, self).__init__(cf, weight, ignore_index)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, inputs, targets):
        loss = self.criterion(inputs, targets.view(-1))
        if self.cf.normalize_loss:
            n, c, h, w = inputs.size()
            loss = loss / (n*h*w)
        return loss
