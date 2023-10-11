import sys
from .Unet import UNet as unet
from torch import nn
import torch

# from .res_unet_adrian import WNet as wnet
class wnet(torch.nn.Module):
    def __init__(self, n_classes=1, in_c=3, layers=(8,16,32), conv_bridge=True, shortcut=True):
        super(wnet, self).__init__()
        self.unet1 = unet(in_c=in_c, n_classes=n_classes, layers=layers, conv_bridge=conv_bridge, shortcut=shortcut)
        self.unet2 = unet(in_c=in_c+n_classes, n_classes=n_classes, layers=layers, conv_bridge=conv_bridge, shortcut=shortcut)
        self.n_classes = n_classes

    def forward(self, x):
        x1 = self.unet1(x)
        x2 = self.unet2(torch.cat([x, x1], dim=1))
        if not self.training:
            return x2
        return x1,x2

class Build_WNet(torch.nn.Module):
    def __init__(self,configs):
        super(Build_WNet, self).__init__()
        self.pos_embed= nn.Parameter(torch.tensor(0.5, requires_grad=True))
        self.backbone=wnet(
             in_c=configs["in_channels"],
            n_classes=configs["num_classes"],
            layers= configs['layer_number'],
        )
    def forward(self,x_pos):
        x,pos=x_pos
        x=x*(1-self.pos_embed)+pos*self.pos_embed
        out=self.backbone(x)
        return out