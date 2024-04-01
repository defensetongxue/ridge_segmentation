import json
import os
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from torch.nn.functional import pad
import torch.nn.functional as F
def generate_segmentation_mask(data_path, patch_size, stride):
    os.makedirs(os.path.join(data_path,'ridge_seg_patchtify'), exist_ok=True)
    # Clean up the directories
    os.makedirs(os.path.join(data_path,'ridge_seg_patchtify','images'), exist_ok=True)
    os.system(f"find {os.path.join(data_path,'ridge_seg_patchtify','images')} -type f -delete")
    os.makedirs(os.path.join(data_path,'ridge_seg_patchtify','masks'), exist_ok=True)
    os.system(f"find {os.path.join(data_path,'ridge_seg_patchtify','masks')} -type f -delete")
    
    data_list=os.listdir(os.path.join(data_path,'images'))
    annotate={}
    cnt=0
    for image_name in data_list:
        mask = Image.open(os.path.join(data_path,'masks',image_name)).convert('L')
        mask=mask.resize((1600,1200),resample=Image.Resampling.NEAREST)
        mask_tensor = torch.from_numpy(np.array(mask, np.float32, copy=False))
        mask_tensor[mask_tensor != 0] = 1
        
        img = Image.open(os.path.join(data_path,'enhanced',image_name)).convert("RGB")
        img=img.resize((1600,1200),resample=Image.Resampling.BILINEAR)
        img_tensor = transforms.ToTensor()(img)
        
        # Calculate padding
        image_size = img_tensor.shape[-2:]
        padding_height = stride - (image_size[0] % stride) if image_size[0] % stride != 0 else 0
        padding_width = stride - (image_size[1] % stride) if image_size[1] % stride != 0 else 0
        
        # Pad image, mask, 
        img_tensor = pad(img_tensor, (0, padding_width, 0, padding_height))
        mask_tensor = mask_tensor.unsqueeze(0)
        mask_tensor = pad(mask_tensor, (0, padding_width, 0, padding_height)).squeeze(0)
        # Create and save cropped patches directly
        for i in range(0, img_tensor.shape[2] - patch_size, stride):  # Change +1 to +patch_size
            for j in range(0, img_tensor.shape[1] - patch_size, stride):  # Change +1 to +patch_size
                img_patch = img_tensor[:, j:j+patch_size, i:i+patch_size]
                mask_patch = mask_tensor[j:j+patch_size, i:i+patch_size]
                save_name = f"{image_name[:-4]}_{str(i//stride)}_{str(j//stride)}"

                img_patch_path = os.path.join(data_path, 'ridge_seg_patchtify', 'images', f"{save_name}.jpg")
                mask_patch_path = os.path.join(data_path, 'ridge_seg_patchtify', 'masks', f"{save_name}.png")

                Image.fromarray((img_patch.permute(1, 2, 0).numpy() * 255).astype(np.uint8)).save(img_patch_path)
                Image.fromarray((mask_patch.numpy() * 255).astype(np.uint8)).save(mask_patch_path)
                
                annotate[save_name] = {
                    "crop_from": image_name,
                    "image_path": img_patch_path,
                    "mask_path": mask_patch_path,
                    "coordinates": (i, j),
                    "patch_size": patch_size,
                    "stride": stride
                }
    
        cnt+=1
    # Save the annotation
    with open(os.path.join(data_path, 'ridge_seg_patchtify', 'annotations.json'), 'w') as f:
        json.dump(annotate, f)
    print("finish patchtify image numebr: ",str(cnt))
if __name__=='__main__':
    from config import get_config
    args=get_config()
    generate_segmentation_mask(args.data_path,args.patch_size,args.stride)