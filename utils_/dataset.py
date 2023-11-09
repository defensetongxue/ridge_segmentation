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
            self.data_list=json.load(f)
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
        data = self.data_list[data_name]
        
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

class ridge_finetone_dataset(Dataset):
    def __init__(self, data_path, split, split_name,configs,model):
        with open(os.path.join(data_path, 'ridge_seg', 'split', f'{split_name}.json'), 'r') as f:
            split_list=json.load(f)
        with open(os.path.join(data_path, 'ridge_seg', 'annotations.json'), 'r') as f:
            self.data_list=json.load(f)
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
        model.eval()
        self._get_finetone_extradata(
            data_path=data_path,
            split_name=split_name,
            split=split,
            configs=configs,
            model=model
        )
    
    def __getitem__(self, idx):
        data_name = self.split_list[idx]
        data = self.data_list[data_name]
        
        img = Image.open(data['image_path']).convert('RGB')
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
        return img, gt, data_name

    def __len__(self):
        return len(self.split_list)
    
    def _get_finetone_extradata(self,data_path,split_name,split,configs,model):
        print("begin to build fine extra_data for "+split)
        with open(os.path.join(data_path,'annotations.json'),'r') as f:
            data_dict=json.load(f)
        with open(os.path.join(data_path,'split',f'{split_name}.json'),'r') as f:
            full_split=json.load(f)[split]
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        img_transforms=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4623, 0.3856, 0.2822],
                std=[0.2527, 0.1889, 0.1334])])
        total_number=0
        for image_name in full_split:
            data=data_dict[image_name]
            if data['stage']>0:
                continue # already in training dataset
            
            img = Image.open(data['image_path']).convert('RGB')
            with torch.no_grad():
                img_tensor = img_transforms(img)

            img=img_tensor.unsqueeze(0).to(device)
            output_img = model(img).squeeze().cpu()
            # Resize the output to the original image size
        
            mask=torch.sigmoid(output_img).cpu()
            if torch.max(mask)>=configs['finetone_threshold_low'] and \
            torch.max(mask)<=configs['finetone_threshold_up']:
                samples_points=self._get_sample_points(mask.detach(),
                                        threshold_low=configs['finetone_threshold_low'],
                                        threshold_up=configs['finetone_threshold_up'],
                                        split=split)
                self._get_extra_sample(
                    data_path=data_path,
                    points=samples_points,
                    data=data,
                    image_name=image_name,
                    patch_size=configs['patch_size'])
                total_number+=len(samples_points)
        print(f"add {total_number} data")
    def _get_extra_sample(self, data_path,points, data,image_name, patch_size):
        image_path = data['enhanced_path']
        img = Image.open(image_path).convert('RGB')

        for sample_cnt, (x, y) in enumerate(points):
            
            save_name = f"{data['id']}_{sample_cnt}"
            img_patch_path = os.path.join(data_path,'ridge_seg', 'finetone', f"{save_name}.jpg")
            mask_patch_path = os.path.join(data_path,'ridge_seg', 'finetone', f"{save_name}.png")

            x_start, x_end = max(0, x-patch_size//2), min(img.width, x+patch_size//2)
            y_start, y_end = max(0, y-patch_size//2), min(img.height, y+patch_size//2)

            img_patch = img.crop((x_start, y_start, x_end, y_end))

            # Calculate padding required
            pad_left = abs(min(0, x-patch_size//2))
            pad_right = max(0, x+patch_size//2 - img.width)
            pad_top = abs(min(0, y-patch_size//2))
            pad_bottom = max(0, y+patch_size//2 - img.height)

            img_patch = ImageOps.expand(img_patch, border=(pad_left, pad_top, pad_right, pad_bottom), fill='black')

            img_patch.save(img_patch_path)

            mask_patch_np = np.zeros((patch_size, patch_size), dtype=np.uint8)
            mask_patch = Image.fromarray(mask_patch_np)
            mask_patch.save(mask_patch_path)

            self.split_list.append(save_name)
            self.data_list[save_name] = {
                    "crop_from": image_name,
                    "image_path": img_patch_path,
                    "mask_path": mask_patch_path,
                    "coordinates": (x,y),
                    "patch_size": patch_size,
                    "stride": 64,
                }
    def _get_sample_points(self, mask, threshold_low=0.4, threshold_up=0.4, split='train'):
        mask_np = mask.numpy()
        sample_points = []
        clear_width = 64
        if split == 'val':
            clear_width = 128
    
        while np.max(mask_np) >= threshold_low and \
            np.max(mask_np)<=threshold_up:
            # Get the coordinate of the maximum value point
            y, x = np.unravel_index(np.argmax(mask_np, axis=None), mask_np.shape)
            sample_points.append((x, y))
    
            # Add the surrounding points
            sample_points.extend([(x+64, y), (x-64, y), (x, y+64), (x, y-64)])
    
            # Set the neighborhood around this point to 0
            x_start, x_end = max(0, x-clear_width), min(mask_np.shape[1], x+clear_width)
            y_start, y_end = max(0, y-clear_width), min(mask_np.shape[0], y+clear_width)
            mask_np[y_start:y_end, x_start:x_end] = 0
    
            # Add 5 random points for balance
            # Make sure the random points are within the mask boundaries
            for _ in range(5):
                rand_x = np.random.randint(0, mask_np.shape[1])
                rand_y = np.random.randint(0, mask_np.shape[0])
                sample_points.append((rand_x, rand_y))
    
        return sample_points
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
    
class ridge_all_dataset(Dataset):
    def __init__(self, data_path, split, split_name):
        with open(os.path.join(data_path,'annotations.json'),'r') as f:
            self.data_dict=json.load(f)
        with open(os.path.join(data_path,'split',f'{split_name}.json'),'r') as f:
            # split_list=json.load(f)[split]
            self.split_list=json.load(f)[split]
        # self.split_list=[]
        # for image_name in split_list:
        #     if 'ridge' in self.data_dict[image_name]:
        #         self.split_list.append(image_name)
        #     else:
        #         if random.random()>zero_sample_rate:
        #             self.split_list.append(image_name)
        self.split = split
        self.resize=transforms.Resize((600,800))
        self.transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            # Fix_RandomRotation(),
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
        
        img = Image.open(data['enhanced_path']).convert('RGB')
        # if 'ridge_diffusion_path' in data:
        #     gt = Image.open(data['ridge_diffusion_path'])
        # else:
        #     gt = Image.new('L', img.size)
        img=self.resize(img)
        # gt=self.resize(gt)
        if self.split == "train":
            seed = torch.seed()
            torch.manual_seed(seed)
            img = self.transforms(img)
            # torch.manual_seed(seed)
            # gt = self.transforms(gt)

        # Convert mask and pos_embed to tensor
        # gt = self.totenor(gt)
        # gt[gt != 0] = 1.
        img = self.img_transforms(img)
        assert img.shape[1]==600,img.shape
        if 'ridge' in data:
            gt=1
        else:
            gt=0
        return img, gt, data_name

    def __len__(self):
        return len(self.split_list)
