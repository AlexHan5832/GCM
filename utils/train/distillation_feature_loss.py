import torch
import torch.nn as nn
import torch.nn.functional as F


class DistillationLoss(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.name = 'distillation_feature'

    def forward(self, feature: list, t_feature: list, **kw):
        sum_loss = 0
        b, _, w, h = feature[0].shape
        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        sum_loss += torch.sum(torch.abs(cos(feature[0], t_feature[0]))) / (w * h)
        sum_loss += torch.sum(torch.abs(cos(feature[1], t_feature[1]))) / (w * h)
        return sum_loss



