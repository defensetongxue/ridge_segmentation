import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from .tools import Fix_RandomRotation
import json
from PIL import Image,ImageEnhance  
import numpy as np

class ridge_segmentataion_dataset(Dataset):
    def __init__(self, data_path, split, split_name):
        with open(os.path.join(data_path, 'ridge_seg', 'split', f'{split_name}.json'), 'r') as f:
            split_list=json.load(f)
        with open(os.path.join(data_path, 'ridge_seg', 'annotations.json'), 'r') as f:
            self.data_list=json.load(f)
        self.split_list=split_list[split]
        self.img_enhance=ContrastEnhancement()
        self.split = split
        self.transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            Fix_RandomRotation(),
        ])
        self.img_transforms=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4623, 0.3856, 0.2822],
                std=[0.2527, 0.1889, 0.1334])])
        self.totenor=transforms.ToTensor()
    def __getitem__(self, idx):
        data_name = self.split_list[idx]
        data = self.data_list[data_name]
        
        img = Image.open(data['image_path']).convert('RGB')
        img = self.img_enhance(img)
        if data['mask_path']:
            gt = Image.open(data['mask_path'])
        else:
            patch_size = data['patch_size']
            gt = Image.new('L', (patch_size, patch_size))  # create a blank (black) image

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
        return img, gt, data

    def __len__(self):
        return len(self.split_list)
    
class ContrastEnhancement:
    def __init__(self, factor=1.5):
        self.factor = factor

    def __call__(self, img):
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(self.factor)
        return img