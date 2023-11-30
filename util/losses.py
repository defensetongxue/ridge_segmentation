import torch
import torch.nn as nn
from timm.loss import LabelSmoothingCrossEntropy

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, ignore_index=255, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.CE_loss = nn.CrossEntropyLoss(
            reduction=reduction, ignore_index=ignore_index, weight=alpha)

    def forward(self, output, target):
        logpt = self.CE_loss(output, target)
        pt = torch.exp(-logpt)
        loss = ((1 - pt) ** self.gamma) * logpt
        if self.reduction == 'mean':
            return loss.mean()
        return loss.sum()


# class BCELoss(nn.Module):
#     def __init__(self, reduction="mean", pos_weight=5.0):
#         pos_weight = torch.tensor(pos_weight).cuda()
#         super(BCELoss, self).__init__()
#         self.bce_loss = nn.BCEWithLogitsLoss(
#             reduction=reduction, pos_weight=pos_weight)
#         self.cls_loss=LabelSmoothingCrossEntropy()
#     def forward(self, prediction, targets):
#         seg_pre,cls_pre=prediction
#         seg_tar,cls_tar=targets
        return self.bce_loss(seg_pre,seg_tar)+self.cls_loss(cls_pre,cls_tar)

class BCELoss(nn.Module):
    def __init__(self, reduction="mean", pos_weight=1.0):
        super(BCELoss, self).__init__()
        self.pos_weight = torch.tensor(pos_weight).cuda()
        self.bce_loss = nn.BCEWithLogitsLoss(
            reduction=reduction, pos_weight=self.pos_weight)
        self.cls_loss = LabelSmoothingCrossEntropy()  # Assuming this is already defined

    def forward(self, prediction, targets):
        seg_pre, cls_pre = prediction
        seg_tar, cls_tar = targets

        # Calculate classification loss
        cls_loss = self.cls_loss(cls_pre, cls_tar)

        # Select instances where cls_tar > 0
        selected_indices = (cls_tar > 0)
        
        # Check if there are any instances to calculate seg_loss
        if selected_indices.sum() > 0:
            seg_pre_selected = seg_pre[selected_indices]
            seg_tar_selected = seg_tar[selected_indices]
            seg_loss = self.bce_loss(seg_pre_selected, seg_tar_selected)
        else:
            # No instances to calculate seg_loss, return zero loss
            seg_loss = 0.0

        # Combine classification and segmentation losses
        total_loss = cls_loss + seg_loss
        return total_loss
    
class CELoss(nn.Module):
    def __init__(self, weight=[1, 1], ignore_index=-100, reduction='mean'):
        super(CELoss, self).__init__()
        weight = torch.tensor(weight).cuda()
        self.CE = nn.CrossEntropyLoss(
            weight=weight, ignore_index=ignore_index, reduction=reduction)

    def forward(self, output, target):
        loss = self.CE(output, target.squeeze(1).long())
        return loss


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-8):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, prediction, target):
        prediction = torch.sigmoid(prediction)
        intersection = 2 * torch.sum(prediction * target) + self.smooth
        union = torch.sum(prediction) + torch.sum(target) + self.smooth
        loss = 1 - intersection / union
        return loss


class CE_DiceLoss(nn.Module):
    def __init__(self, reduction="mean", D_weight=0.5):
        super(CE_DiceLoss, self).__init__()
        self.DiceLoss = DiceLoss()
        self.BCELoss = BCELoss(reduction=reduction)
        self.D_weight = D_weight

    def forward(self, prediction, targets):
        return self.D_weight * self.DiceLoss(prediction, targets) + (1 - self.D_weight) * self.BCELoss(prediction,
                                                                                                       targets)

class WNetLoss(nn.Module):
    def __init__(self, reduction="mean", pos_weight=1.0):
        pos_weight = torch.tensor(pos_weight).cuda()
        super(WNetLoss, self).__init__()
        self.bce_loss = nn.BCEWithLogitsLoss(
            reduction=reduction, pos_weight=pos_weight)

    def forward(self, prediction, targets):
        if isinstance(prediction,tuple):
            return +self.bce_loss(prediction[0],targets) + \
                self.bce_loss(prediction[1],targets)
        return self.bce_loss(prediction, targets)