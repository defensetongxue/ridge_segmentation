import os,random
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from .tools import Fix_RandomRotation
import json
from PIL import Image,ImageOps
import numpy as np
class ridge_segmentataion_dataset(Dataset):
    def __init__(self, data_path, split, split_name):
        with open(os.path.join(data_path, 'ridge_seg', 'split', f'{split_name}.json'), 'r') as f:
            split_list=json.load(f)
        with open(os.path.join(data_path, 'ridge_seg', 'annotations.json'), 'r') as f:
            self.data_dict=json.load(f)
        self.split_list=split_list[split]
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
        data = self.data_dict[data_name]
        
        img = Image.open(data['image_path']).convert('RGB')
        if data['mask_path']:
            gt = Image.open(data['mask_path'])
        else:
            raise
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

class ridge_trans_dataset(Dataset):
    def __init__(self, data_path, split, split_name):
        with open(os.path.join(data_path, 'split', f'{split_name}.json'), 'r') as f:
            split_list=json.load(f)[split]
        with open(os.path.join(data_path, 'annotations.json'), 'r') as f:
            self.data_dict=json.load(f)
        if split=='train':
            ridge_list=[]
            for image_name in split_list:
                if 'ridge' in self.data_dict[image_name]:
                    ridge_list.append(image_name)
                    
            self.split_list=ridge_list
        else:
            self.split_list=split_list
        self.split = split
        self.resize=transforms.Resize((512,512))
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
        image_name = self.split_list[idx]
        data = self.data_dict[image_name]
        
        img = Image.open(data['image_path']).convert('RGB')
        if 'ridge_diffusion_path' in data:
            gt = Image.open(data['ridge_diffusion_path'])
        else:
            gt = Image.new('L', img.size)
        img=self.resize(img)
        gt=self.resize(gt)
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
        if self.split=='val':
            ridge_label=1 if 'ridge' in data else 0
            return img,ridge_label,image_name
        return img, gt, image_name

    def __len__(self):
        return len(self.split_list)
    
class ridege_finetone_val(Dataset):
    def __init__(self,data_path,split_name,split) :
        super().__init__()
        with open(os.path.join(data_path,'annotations.json'),'r') as f:
            self.data_dict=json.load(f)
        with open(os.path.join(data_path,'split',f'{split_name}.json'),'r') as f:
            self.split_list=json.load(f)[split]
        assert split !='train'
        self.img_transforms=transforms.Compose([
            transforms.Resize((600,800)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4623, 0.3856, 0.2822],
                std=[0.2527, 0.1889, 0.1334])])
    def __len__(self):
        return len(self.split_list)
    def __getitem__(self, idx):
        image_name=self.split_list[idx]
        data=self.data_dict[image_name]
        img = Image.open(data['enhanced_path']).convert('RGB')
        if 'ridge' not in data:
            label=0
        else:
            label=1
        img=self.img_transforms(img)
        return img,label,data['stage']
    