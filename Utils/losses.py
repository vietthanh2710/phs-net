import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, y_pred, y_true):
        y_pred = torch.sigmoid(y_pred)
        y_pred = y_pred.view(-1)
        y_true = y_true.view(-1)
        intersection = (y_pred * y_true).sum()
        return 1 - (2. * intersection + self.smooth) / (y_pred.sum() + y_true.sum() + self.smooth)

class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.7, smooth=1e-5):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.smooth = smooth

    def forward(self, y_pred, y_true):
        y_pred = torch.sigmoid(y_pred)
        y_true_pos = y_true.view(-1)
        y_pred_pos = y_pred.view(-1)
        true_pos = torch.sum(y_true_pos * y_pred_pos)
        false_neg = torch.sum(y_true_pos * (1 - y_pred_pos))
        false_pos = torch.sum((1 - y_true_pos) * y_pred_pos)
        return 1 - (true_pos + self.smooth) / (
            true_pos + self.alpha * false_neg + (1 - self.alpha) * false_pos + self.smooth
        )

class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha=0.7, gamma=0.75, smooth=1e-5):
        super(FocalTverskyLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, y_pred, y_true):
        y_pred = torch.sigmoid(y_pred)
        y_true_pos = y_true.view(-1)
        y_pred_pos = y_pred.view(-1)
        true_pos = torch.sum(y_true_pos * y_pred_pos)
        false_neg = torch.sum(y_true_pos * (1 - y_pred_pos))
        false_pos = torch.sum((1 - y_true_pos) * y_pred_pos)
        tversky = (true_pos + self.smooth) / (
            true_pos + self.alpha * false_neg + (1 - self.alpha) * false_pos + self.smooth
        )
        return torch.pow((1 - tversky), self.gamma)

class BCETverskyLoss(nn.Module):
    def __init__(self, bce_weight=0.5, alpha=0.7, smooth=1e-5):
        super(BCETverskyLoss, self).__init__()
        self.bce_weight = bce_weight
        self.tversky = TverskyLoss(alpha=alpha, smooth=smooth)

    def forward(self, y_pred, y_true):
        bce = F.binary_cross_entropy_with_logits(torch.sigmoid(y_pred), y_true)
        tv = self.tversky(y_pred, y_true)
        return bce * self.bce_weight + tv * (1 - self.bce_weight)

class DiceTverskyLoss(nn.Module):
    def __init__(self, dice_weight=0.5, alpha=0.7, smooth=1e-5):
        super(DiceTverskyLoss, self).__init__()
        self.dice_weight = dice_weight
        self.dice = DiceLoss(smooth=smooth)
        self.tversky = TverskyLoss(alpha=alpha, smooth=smooth)

    def forward(self, y_pred, y_true):
        dice = self.dice(y_pred, y_true)
        tv = self.tversky(y_pred, y_true)
        return dice * self.dice_weight + tv * (1 - self.dice_weight)