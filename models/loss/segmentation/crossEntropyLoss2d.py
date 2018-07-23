import torch.nn.functional as F
import torch.nn as nn
import torch
from semantic_loss import Semantic_Loss
from torch.autograd import Variable


class CrossEntropyLoss2d(Semantic_Loss):
    def __init__(self, cf, weight=None, ignore_index=255):
        super(CrossEntropyLoss2d, self).__init__(cf, weight, ignore_index)
        self.dropout_layer = nn.Dropout2d(p=0.3)

    def forward(self, inputs, targets):
        # print(inputs.size())
        # n, c, h, w = inputs.size()
        #
        #
        # # log_p = self.my_softmax(inputs).log()
        # log_p = F.log_softmax(inputs[0],dim=1)
        #
        # log_p = log_p.transpose(0, 2).transpose(2, 3).contiguous()                      # I dont understand this part...
        # log_p = log_p[targets.view(n, h, w, 1).repeat(1, 1, 1, c) != self.ignore_index] # rm invalid index
        # log_p = log_p.view(-1, c)
        #
        # mask = (targets != self.ignore_index)
        # targets = targets[mask]
        #
        # targets = targets.view(-1)
        #
        # loss = F.nll_loss(log_p, targets, weight=None, size_average=False)
        #
        # if self.cf.normalize_loss:
        #    loss /= mask.data.sum()
        loss_fn_ = torch.nn.NLLLoss2d(weight=None, size_average=True,
                                      ignore_index=self.ignore_index)

        loss = loss_fn_(F.log_softmax(inputs), targets)
        return loss#.mean()

    def my_softmax(self, inputs):

        d,n,c,w,h = inputs.size()

        inputs = inputs.transpose(2,4).contiguous().view(-1,c)
        max_indexes = torch.max(inputs, 1)[1]

        data_size = max_indexes.size(0)
        mask_indexes = Variable(torch.FloatTensor(data_size,c),requires_grad=True).cuda()
        mask_indexes =  mask_indexes.scatter_(1,max_indexes.view(-1,1),1)
        inputs = mask_indexes.view(d,n,h,w,c).transpose(4,2)

        inputs = inputs.sum(dim=0)
        inputs = inputs.float()
        probs = inputs/d

        probs = torch.clamp(probs,min=1e-20,max=1)


        return probs
