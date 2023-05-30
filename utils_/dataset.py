import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from .tools import Fix_RandomRotation
import json
from PIL import Image
import numpy as np

class ridge_segmentataion_dataset(Dataset):
    def __init__(self, data_path, split):
        with open(os.path.join(data_path, 'ridge_seg', 'annotations', f'{split}.json'), 'r') as f:
            self.annote=json.load(f)
        self.split=split
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
        data = self.annote[idx]

        # Read the image and mask
        img = Image.open(data['img_path']).convert('RGB')
        gt = Image.open(data['mask_path'])

        # Transform mask back to 0,1 tensor
        gt = torch.from_numpy(np.array(gt, np.int32, copy=False))
        gt[gt != 0] = 1

        if self.split == "train" :
            seed = torch.seed()
            torch.manual_seed(seed)
            img = self.transforms(img)
            torch.manual_seed(seed)
            gt = self.transforms(gt)

        img = self.img_transforms(img)

        return img, gt,data['class']

    def __len__(self):
        return len(self.annote)
