import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import resnest.torch as resnest_torch


class BCEWithLogitsLoss_LabelSmooth(nn.Module):
    def __init__(self, label_smoothing=0.1, pos_weight=None):
        super().__init__()
        self.label_smoothing = label_smoothing
        self.pos_weight      = pos_weight

    def forward(self, input, target):
        target_smoothed = target.float() * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
        if self.pos_weight is not None:
            loss = F.binary_cross_entropy_with_logits(input, target_smoothed.type_as(input),
                                                      pos_weight=torch.tensor(self.pos_weight))
        else:
            loss = F.binary_cross_entropy_with_logits(input, target_smoothed.type_as(input))
        return loss


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=1.5, logits=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="mean")
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduction="mean")
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        return F_loss


class PANNsLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCELoss()
        self.bce_logit = nn.BCEWithLogitsLoss()  # BCEWithLogitsLoss more stable

    def forward(self, input, target):
        input_ = input["clipwise_output"]  # sigmoid
        input_ = torch.where(torch.isnan(input_), torch.zeros_like(input_), input_)
        input_ = torch.where(torch.isinf(input_), torch.zeros_like(input_), input_)

        target = target.float()
        # return self.bce_logit(input_, target)
        return self.bce(input_, target)