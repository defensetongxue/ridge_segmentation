import torch
from torch import nn 
from .unet_torch import UNet
class lwnet_torch(nn.Module):
    def __init__(self, configs):
        super(lwnet_torch, self).__init__()
        self.unet1=UNet(n_channels=configs["in_channels"],
                       n_classes=configs["num_classes"])
        self.unet2=UNet(n_channels=configs["in_channels"]+configs["num_classes"],
                       n_classes=configs["num_classes"])
        self.unet1._init_weight(configs["pretrained"])
        self.unet2._init_weight(configs["pretrained"])
        self.pos_embed= nn.Parameter(torch.tensor(0.5, requires_grad=True))

    def forward(self,x_pos):
        x,pos=x_pos
        x=x*(1-self.pos_embed)+pos*self.pos_embed
        out1=self.unet1(x)
        x=torch.cat([x,out1],dim=1)
        out2=self.unet2(x)
        if self.training:
            return out1,out2
        return out2