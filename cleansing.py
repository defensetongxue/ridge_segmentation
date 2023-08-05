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
def generate_segmentation_mask(data_path, patch_size, stride,split_name='0'):
    # Generate path_image folder
    os.makedirs(os.path.join(data_path,'ridge_seg','images'),exist_ok=True)
    os.makedirs(os.path.join(data_path,'ridge_seg','masks'),exist_ok=True)
    os.makedirs(os.path.join(data_path,'ridge_seg','annotations'),exist_ok=True)
    os.makedirs(os.path.join(data_path,'ridge_seg','position_embed'),exist_ok=True)
    
    os.system(f"find {os.path.join(data_path,'ridge_seg','images')} -type f -delete")
    os.system(f"find {os.path.join(data_path,'ridge_seg','masks')} -type f -delete")
    os.system(f"find {os.path.join(data_path,'ridge_seg','annotations')} -type f -delete")
    os.system(f"find {os.path.join(data_path,'ridge_seg','position_embed')} -type f -delete")

    with open(os.path.join(data_path,'split',f"{split_name}.json"),'r')  as f:
        split_loaded=json.load(f)
        
    with open(os.path.join(data_path,'annotations.json'),'r') as f:
        data_list=json.load(f)
    splits=['train','val']
    for split_name in splits:
        split_list=split_loaded[split_name]
        annotate=[]
        print(f"paser {split_name} data begining, there are {len(data_list)} data")
        cnt=0
        for  image_name in split_list:
            cnt+=1
            if cnt % 100==0:
                print(f"Finish number: {cnt}")
            data=data_list[image_name]
            if 'ridge' in data: # this image has ridge annotation
                mask = Image.open(data['ridge_diffusion_path'])
                mask_tensor=torch.from_numpy(np.array(mask,np.float32, copy=False))
                mask_tensor[mask_tensor!=0]=1

                # Load image, position_embed and mask
                img = Image.open(data['image_path']).convert("RGB")
                img_tensor = transforms.ToTensor()(img)

                pos_embed=torch.load(data['pos_embed_path'])
                pos_embed=F.interpolate(pos_embed[None,None,:,:], size=mask_tensor.shape, mode='nearest')
                pos_embed=pos_embed.squeeze()
                
                # Image size
                image_size = img_tensor.shape[-2:]

                # Calculate padding
                padding_height = stride - (image_size[0] % stride) if image_size[0] % stride != 0 else 0
                padding_width = stride - (image_size[1] % stride) if image_size[1] % stride != 0 else 0

                # Pad image
                img_tensor = pad(img_tensor, (0, padding_width, 0, padding_height))

                # Pad mask and pos_embed, adjust dimensions to CxHxW for padding
                mask_tensor = mask_tensor.unsqueeze(0) # adjust mask tensor to CxHxW
                mask_tensor = pad(mask_tensor, (0, padding_width, 0, padding_height)).squeeze(0) # apply padding and revert back to HxW

                pos_embed = pos_embed.unsqueeze(0) # adjust pos_embed tensor to CxHxW
                pos_embed = pad(pos_embed, (0, padding_width, 0, padding_height)).squeeze(0) # apply padding and revert back to HxW

                # Unfold to patches
                patches_img = img_tensor.unfold(1, patch_size, stride).unfold(2, patch_size, stride)
                patches_mask = mask_tensor.unfold(0, patch_size, stride).unfold(1, patch_size, stride)
                pos_embed_patches=pos_embed.unfold(0, patch_size, stride).unfold(1, patch_size, stride) 
                
                # Loop through all patches
                for i in range(patches_img.shape[1]):
                    for j in range(patches_img.shape[2]):
                        patch_img = patches_img[:,i,j].permute(1, 2, 0).numpy()
                        patch_mask = patches_mask[i,j].numpy()
                        pos_embed_patch=pos_embed_patches[i,j].numpy()
                        # Save image patch
                        patch_name=f"{data['image_name'].split('.')[0]}_{i}_{j}"
                        img_path = os.path.join(data_path, 'ridge_seg', 'images', f"{patch_name}.jpg")
                        Image.fromarray((patch_img * 255).astype(np.uint8)).save(img_path)

                        # Save mask patch
                        mask_path = os.path.join(data_path, 'ridge_seg', 'masks', f"{patch_name}.png")
                        Image.fromarray((patch_mask * 255).astype(np.uint8)).save(mask_path)

                        # Save pos embed
                        pos_embed_path=os.path.join(data_path,'ridge_seg','position_embed',f'{patch_name}.png')
                        Image.fromarray((pos_embed_patch* 255).astype(np.uint8)).save(pos_embed_path)
                        # Get class
                        class_annote = data["class"] if patch_mask.max() > 0 else 0

                        annotate.append({
                            "img_path": img_path,
                            "mask_path": mask_path,
                            'pos_embed_path':pos_embed_path,
                            "class": class_annote
                        })
            else:
                
                img = Image.open(data['image_path']).convert("RGB")
                img_tensor = transforms.ToTensor()(img)

                pos_embed=torch.load(data['pos_embed_path'])
                pos_embed=F.interpolate(pos_embed[None,None,:,:], size=img_tensor.shape[-2:], mode='nearest')
                pos_embed=pos_embed.squeeze()
                
                patches_img = img_tensor.unfold(1, patch_size, stride).unfold(2, patch_size, stride)
                pos_embed_patches=pos_embed.unfold(0, patch_size, stride).unfold(1, patch_size, stride) 
                # Image size
                image_size = img_tensor.shape[-2:]

                # Calculate padding
                padding_height = stride - (image_size[0] % stride) if image_size[0] % stride != 0 else 0
                padding_width = stride - (image_size[1] % stride) if image_size[1] % stride != 0 else 0

                # Pad image
                img_tensor = pad(img_tensor, (0, padding_width, 0, padding_height))

                # Pad and pos_embed, adjust dimensions to CxHxW for padding
                
                pos_embed = pos_embed.unsqueeze(0) # adjust pos_embed tensor to CxHxW
                pos_embed = pad(pos_embed, (0, padding_width, 0, padding_height)).squeeze(0) # apply padding and revert back to HxW
                # Unfold to patches
                patches_img = img_tensor.unfold(1, patch_size, stride).unfold(2, patch_size, stride)
                pos_embed_patches=pos_embed.unfold(0, patch_size, stride).unfold(1, patch_size, stride) 
                
                for i in range(patches_img.shape[1]):
                    for j in range(patches_img.shape[2]):
                        patch_img = patches_img[:,i,j].permute(1, 2, 0).numpy()
                        pos_embed_patch=pos_embed_patches[i,j].numpy()
                        # Save image patch
                        patch_name=f"{data['image_name'].split('.')[0]}_{i}_{j}"
                        img_path = os.path.join(data_path, 'ridge_seg', 'images', f"{patch_name}.jpg")
                        Image.fromarray((patch_img * 255).astype(np.uint8)).save(img_path)

                        # Save mask patch
                        mask_path = os.path.join(data_path, 'ridge_seg', 'masks', f"{patch_name}.png")
                        Image.fromarray((np.zeros_like(pos_embed_patch)).astype(np.uint8)).save(mask_path)

                        # Save pos embed
                        pos_embed_path=os.path.join(data_path,'ridge_seg','position_embed',f'{patch_name}.png')
                        Image.fromarray((pos_embed_patch* 255).astype(np.uint8)).save(pos_embed_path)
                        # Get class
                        class_annote = 0

                        annotate.append({
                            "img_path": img_path,
                            "mask_path": mask_path,
                            'pos_embed_path':pos_embed_path,
                            "class": class_annote
                        })
        with open(os.path.join(data_path, 'ridge_seg', 'annotations', f'{split}.json'), 'w') as f:
            json.dump(annotate, f)


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