import os,random
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from .tools import Fix_RandomRotation
import json
from PIL import Image,ImageOps
import numpy as np
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
class ridge_segmentataion_dataset(Dataset):
    def __init__(self, data_path, split,factor=0.25):
        with open(os.path.join(data_path, 'ridge_seg_patchtify', 'annotations.json'), 'r') as f:
            self.data_dict=json.load(f)
        
        self.split_list=list(self.data_dict.keys())
        self.split = split
        self.transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            Fix_RandomRotation(),
        ])
        if factor!=1.:
            self.mask_resize=transforms.Resize((100,100), interpolation=transforms.InterpolationMode.NEAREST)
        else:
            self.mask_resize=None
        self.img_transforms=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=IMAGENET_DEFAULT_MEAN,std=IMAGENET_DEFAULT_STD
            )])
        self.totenor=transforms.ToTensor()
    def __getitem__(self, idx):
        data_name = self.split_list[idx]
        data = self.data_dict[data_name]
        
        img = Image.open(data['image_path']).convert('RGB')
        if data['mask_path']:
            gt = Image.open(data['mask_path'])
            if self.mask_resize:
                gt=self.mask_resize(gt)

        if self.split == "train":
            seed = torch.seed()
            torch.manual_seed(seed)
            img = self.transforms(img)
            torch.manual_seed(seed)
            gt = self.transforms(gt)

        # Convert mask and pos_embed to tensor
        gt = self.totenor(gt)
        gt[gt != 0] = 1.
        img = self.img_transforms(img)
        return img, gt, data_name

    def __len__(self):
        return len(self.split_list)
