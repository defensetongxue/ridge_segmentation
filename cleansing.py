import json
import os
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from torch.nn.functional import pad
import torch.nn.functional as F
import sys
sys.path.append('..')
from ROP_diagnoise import generate_ridge_diffusion
from ROP_diagnoise import generate_ridge
def generate_segmentation_mask(data_path, patch_size, stride):
    # Generate path_image folder
    os.makedirs(data_path,'ridge_seg',exist_ok=True)
    # save the padded image here
    os.makedirs(os.path.join(data_path,'ridge_seg','images'),exist_ok=True)
    os.system(f"find {os.path.join(data_path,'ridge_seg','images')} -type f -delete")
    # save the padded mask here
    os.makedirs(os.path.join(data_path,'ridge_seg','masks'),exist_ok=True)
    os.system(f"find {os.path.join(data_path,'ridge_seg','masks')} -type f -delete")
    # save the padded pos_embed here
    os.makedirs(os.path.join(data_path,'ridge_seg','position_embed'),exist_ok=True)
    os.system(f"find {os.path.join(data_path,'ridge_seg','position_embed')} -type f -delete")
    # save the annotation in such format
    '''
    {
        'crop_from',
        'image_path',
        'pos_embed_path',
        'mask_path:,
        'left_top':coordinate for the patch's left_top
        'pacth_size':
        'stride':
    }
    '''
    
        
    with open(os.path.join(data_path,'annotations.json'),'r') as f:
        data_list=json.load(f)
    
    annotate={}
    cnt=0
    for image_name in data_list:
        cnt+=1
        if cnt % 100==0:
            print(f"Finish number: {cnt}")
        data=data_list[image_name]

        # Load image, position_embed and mask
        img = Image.open(data['image_path']).convert("RGB")
        img_tensor = transforms.ToTensor()(img)
        mask = Image.open(data['ridge_diffusion_path'])
        mask_tensor=torch.from_numpy(np.array(mask,np.float32, copy=False))
        mask_tensor[mask_tensor!=0]=1
        pos_embed=torch.load(data['pos_embed_path'])
        pos_embed=F.interpolate(pos_embed[None,None,:,:], size=mask_tensor.shape, mode='nearest')
        pos_embed=pos_embed.squeeze()
    
        # Calculate padding
        image_size = img_tensor.shape[-2:]
        padding_height = stride - (image_size[0] % stride) if image_size[0] % stride != 0 else 0
        padding_width = stride - (image_size[1] % stride) if image_size[1] % stride != 0 else 0
        
        # Pad image, mask and pos_embed
        img_tensor = pad(img_tensor, (0, padding_width, 0, padding_height))
        mask_tensor = mask_tensor.unsqueeze(0) # adjust mask tensor to CxHxW
        mask_tensor = pad(mask_tensor, (0, padding_width, 0, padding_height)).squeeze(0) # apply padding and revert back to HxW
        pos_embed = pos_embed.unsqueeze(0) # adjust pos_embed tensor to CxHxW
        pos_embed = pad(pos_embed, (0, padding_width, 0, padding_height)).squeeze(0) # apply padding and revert back to HxW
        
        # Save the padded image
        padded_img_path = os.path.join(data_path, 'ridge_seg', 'images', f"{data['image_name'].split('.')[0]}_padded.jpg")
        Image.fromarray((img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)).save(padded_img_path)
        
        # Save padded pos embed
        padded_pos_embed_path = os.path.join(data_path, 'ridge_seg', 'position_embed', f'{data["image_name"].split(".")[0]}_padded.jpg')
        Image.fromarray((pos_embed.numpy() * 255).astype(np.uint8)).save(padded_pos_embed_path)
        
        # If 'ridge' is in data, save the padded mask
        if 'ridge' in data:
            padded_mask_path = os.path.join(data_path, 'ridge_seg', 'masks', f"{data['image_name'].split('.')[0]}_padded.png")
            Image.fromarray((mask_tensor.numpy() * 255).astype(np.uint8)).save(padded_mask_path)
        else:
            padded_mask_path = None
        
        # Calculate the number of patches in both directions
        num_patches_height = (img_tensor.shape[1] - patch_size) // stride + 1
        num_patches_width = (img_tensor.shape[2] - patch_size) // stride + 1
        
        # Loop through all patches
        for i in range(num_patches_height):
            for j in range(num_patches_width):
                # Calculate coordinates for the patch's left-top point
                left_top_coordinate = (i * stride, j * stride)
        
        
                annotate[f"{data['id']}_{str(i)}_{str(j)}"]={
                    "crop_from":image_name,
                    "image_path": padded_img_path,
                    "pos_embed_path": padded_pos_embed_path,
                    "mask_path": padded_mask_path,
                    "coordinates": left_top_coordinate,
                    "patch_size": patch_size,
                    "stride": stride,
                }
        
        # Save the annotation
    with open(os.path.join(data_path, 'ridge_seg', 'annotations.json'), 'w') as f:
        json.dump(annotate, f)

def generate_split(data_path,split_name):
    '''generate patch split from orignal split '''
    os.makedirs(data_path,'ridge_seg','split',exist_ok=True)

    with open(os.path.join(data_path,'split',f"{split_name}.json"),'r') as f:
        orignal_split=json.load(f)

    # buid_split_dict image_id: train or val or test.py
    split_dict={}
    for split in ['train','val']:
        # no need to test in training loop, test should be done in visual function in test.py
        image_name_list =orignal_split[split]
        for image_name in image_name_list:
            split_dict[image_name.split('.')[0]]=split # id : split
    
    with open(os.path.join(data_path, 'ridge_seg', 'annotations.json'), 'r') as f:
        patch_data_list=json.load(f)
    new_split={
        'train':[],
        'val':[]
    }
    for data_name in patch_data_list:
        data_id,_,_=data_name.split('_')
        new_split[split[data_id]].append(data_name)
    with open(data_path,'ridge_seg','split',f"{split_name}.json",'r') as f:
        json.dump(new_split,f)     
if __name__=='__main__':
    from config import get_config
    args=get_config()
    
    # cleansing
    if args.generate_ridge:
        generate_ridge(args.json_file_dict,args.path_tar)
        print(f"generate ridge_coordinate in {os.path.join(args.path_tar,'ridge')}")
    if args.generate_diffusion_mask:
        print("begin generate diffusion map")
        generate_ridge_diffusion(args.path_tar)
        print("finished")
    generate_segmentation_mask(args.path_tar,args.patch_size,args.stride)