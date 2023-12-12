import torch
import torch.nn as nn
import torch.nn.functional as F

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


class BCELoss(nn.Module):
    def __init__(self, reduction="mean", pos_weight=1.0):
        pos_weight = torch.tensor(pos_weight).cuda()
        super(BCELoss, self).__init__()
        self.bce_loss = nn.BCEWithLogitsLoss(
            reduction=reduction, pos_weight=pos_weight)

    def forward(self, prediction, targets):
        return self.bce_loss(prediction, targets)


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

class StructureLoss(nn.Module):
    def __init__(self):
        super(StructureLoss,self).__init__()
    def forward(self,pred,mask):
        weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
        wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
        wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))
    
        pred = torch.sigmoid(pred)
        inter = ((pred * mask) * weit).sum(dim=(2, 3))
        union = ((pred + mask) * weit).sum(dim=(2, 3))
        wiou = 1 - (inter + 1) / (union - inter + 1)
    
        return (wbce + wiou).mean()

class Unetv2Loss(nn.Module):
    def __init__(self):
        super(Unetv2Loss,self).__init__()
        self.structure_loss=StructureLoss()
    def forward(self,preds_lists,mask):
        P1, P2=preds_lists
        loss_P1=self.structure_loss(P1,mask)
        loss_p2=self.structure_loss(P2,mask)
        return loss_P1+loss_p2
        