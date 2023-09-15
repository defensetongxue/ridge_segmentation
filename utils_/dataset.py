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

    def __getitem__(self, idx):
        data_name = self.split_list[idx]
        data=self.data_list[data_name]
        # Read the padded image and position embedding
        img = Image.open(data['image_path']).convert('RGB')
        pos_embed = Image.open(data['pos_embed_path'])
        img=self.img_enhance(img)
        # Crop the patch
        left_top_coordinate = data['coordinates']
        patch_size = data['patch_size']
        img = img.crop((left_top_coordinate[0], left_top_coordinate[1], left_top_coordinate[0] + patch_size, left_top_coordinate[1] + patch_size))
        pos_embed = pos_embed.crop((left_top_coordinate[0], left_top_coordinate[1], left_top_coordinate[0] + patch_size, left_top_coordinate[1] + patch_size))

        # Read and crop the mask if it exists
        if data['mask_path']:
            gt = Image.open(data['mask_path'])
            gt = gt.crop((left_top_coordinate[0], left_top_coordinate[1], left_top_coordinate[0] + patch_size, left_top_coordinate[1] + patch_size))
        else:
            gt = Image.new('L', (patch_size, patch_size))  # create a blank (black) image

        if self.split == "train":
            seed = torch.seed()
            torch.manual_seed(seed)
            img = self.transforms(img)
            torch.manual_seed(seed)
            pos_embed = self.transforms(pos_embed)
            torch.manual_seed(seed)
            gt = self.transforms(gt)

        # Transform mask back to 0,1 tensor
        gt = torch.from_numpy(np.array(gt, np.float32, copy=False))
        gt[gt != 0] = 1.
        pos_embed = torch.from_numpy(np.array(pos_embed, np.float32, copy=False))
        img = self.img_transforms(img)

        return (img, pos_embed), gt.squeeze(), data

    def __len__(self):
        return len(self.split_list)
    
class ContrastEnhancement:
    def __init__(self, factor=1.5):
        self.factor = factor

    def __call__(self, img):
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(self.factor)
        return img