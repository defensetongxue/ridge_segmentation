import os
import logging
from .layers import Block

import numpy as np

from functools import partial
import torch
import torch.nn as nn
import torch._utils

BatchNorm2d = nn.BatchNorm2d
BN_MOMENTUM = 0.01
logger = logging.getLogger(__name__)

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class UpBlock(nn.Module):
    def __init__(self, in_channels,out_channels) -> None:
        super(UpBlock,self).__init__()
        self.up_sample=nn.Upsample(mode='bilinear', scale_factor=2, align_corners=False)
        self.up_conv=nn.Conv2d(in_channels, out_channels, kernel_size=1)
        downsample = nn.Sequential(
                nn.Conv2d(2*out_channels, out_channels, kernel_size=1, stride=1, bias=False),
                BatchNorm2d(out_channels, momentum=BN_MOMENTUM)
            )
        self.conv=BasicBlock(2*out_channels,out_channels,downsample=downsample)

    def forward(self,x,skip):
        x=self.up_sample(x)
        x=self.up_conv(x)
        
        x=torch.cat([x,skip],dim=1)
        x=self.conv(x)
        return x
class Transformer(nn.Module):
    def __init__(self,embed_dim=1024, depth=24,
                 num_heads=16, mlp_ratio=4., qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., num_patches=100, norm_layer=partial(nn.LayerNorm, eps=1e-6) ) :
        super(Transformer,self).__init__()
        
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches , embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.norm=norm_layer(embed_dim)
    def forward(self,x):
        bc,embed_dim,h,w=x.shape
        x=x.reshape(bc,embed_dim,-1)
        x=x.transpose(-1,-2)
        x=x+self.pos_embed
        x=self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        x=x.transpose(-1,-2)
        x=x.reshape(bc,embed_dim,h,w)
        return x
    
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        # Initial downsampling
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)

        # Additional downsampling layers
        self.encoder_blocks=nn.ModuleList([
            self._make_layer(BasicBlock,64,128, num_blocks=2, stride=2),
            self._make_layer(BasicBlock, 128,256, num_blocks=2, stride=2),
            self._make_layer(BasicBlock, 256,512, num_blocks=2, stride=2),
            self._make_layer(BasicBlock, 512,1024, num_blocks=2, stride=2) # 1024,
        ])
        self.transformer=Transformer()
        self.decoder_blocks=nn.ModuleList([
            UpBlock(1024,512),
            UpBlock(512,256),
            UpBlock(256,128),
            UpBlock(128,64)
        ])
        self.seghead=self.last_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=1,
                stride=1,
                padding=0),
            BatchNorm2d(64, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=64,
                out_channels=1,
                kernel_size=1,
                stride=1,
                padding=0)
        )
        self.map_size=nn.Upsample(mode='bilinear', scale_factor=4, align_corners=False)
        self.load_pretrained('./RETFound_cfp_weights.pth')
    def load_pretrained(self,attention_weight_path):
        
        checkpoint = torch.load(attention_weight_path, map_location='cpu')
        state_dict = checkpoint['model']
        nn.init.trunc_normal_(self.transformer.pos_embed, std=.02)
        model_state_dict = self.transformer.state_dict()
        for name, param in state_dict.items():
            if name in model_state_dict:
                if param.shape != model_state_dict[name].shape:
                    print(f"Shape mismatch at: {name}, model: {model_state_dict[name].shape}, loaded: {param.shape}")
                else:
                    model_state_dict[name].copy_(param)
                    print(f"Successfully load the param {name}")
    
    def _make_layer(self, block,in_channels, out_channels, num_blocks, stride):
        downsample = None

        if stride != 1 or in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                BatchNorm2d(out_channels * block.expansion, momentum=BN_MOMENTUM)
            )

        layers = []
        layers.append(block(in_channels, out_channels, stride, downsample))
        in_channels = out_channels * block.expansion
        for _ in range(num_blocks-1):
            layers.append(block(in_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # Initial downsampling
        x = self.relu(self.bn1(self.conv1(x))) 
        x = self.relu(self.bn2(self.conv2(x)))
        # bc,64,150,200
        # Additional downsampling
        skip_connection=[x]
        for encoder_block in self.encoder_blocks:
            x = encoder_block(x) 
            skip_connection.append(x)
        # middle stage
        x=self.transformer(x)
        # decoder 
        skip_connection.reverse()
        for i,decoder_block in enumerate(self.decoder_blocks):
            x=decoder_block(x,skip_connection[i+1])
        x=self.seghead(x)
        x=self.map_size(x)
        return x
model=UNet()
inputs=torch.randn((1,3,640,640))
outputs=model(inputs)
print(outputs.shape)