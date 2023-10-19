import os
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
    def _init_weight(self, pretrained_path):
        if os.path.isfile(pretrained_path):
            # Load the pre-trained weights
            pretrained_weights = torch.load(pretrained_path, map_location='cpu')['model_state_dict']

            model_dict = self.state_dict()

            # Check for size mismatch and load weights accordingly
            load_weights = {}
            for name, param in pretrained_weights.items():
                if name in model_dict:
                    if model_dict[name].size() == param.size():
                        load_weights[name] = param
                    else:
                        print(f"Size mismatch, skipping: {name}")
                else:
                    print(f"Parameter not in model, skipping: {name}")

            # Update the model's state dict
            model_dict.update(load_weights)
            self.load_state_dict(model_dict)
        else:
            print("No file found at pretrained_path")
class Build_WNet(torch.nn.Module):
    def __init__(self,configs):
        super(Build_WNet, self).__init__()
        self.backbone=wnet(
             in_c=configs["in_channels"],
            n_classes=configs["num_classes"],
            layers= configs['layer_number'],
        )
        # self.backbone._init_weight(configs['pretrained'])
    
    def forward(self,x):
        out=self.backbone(x)
        return out