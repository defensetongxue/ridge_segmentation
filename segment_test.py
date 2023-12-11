import json
import os
import torch
from config import get_config
from torchvision import transforms
from util import get_instance,k_max_values_and_indices
import models
from PIL import Image, ImageDraw,ImageFont
import numpy as np
# Parse arguments
import time
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
def plot_points_on_image(image_path, points, save_path=None):
    # Open and resize the image
    img = Image.open(image_path).resize((800, 600))
    draw = ImageDraw.Draw(img)

    # Optional: Load a font if you want to use specific font styles
    try:
        font = ImageFont.truetype("arial.ttf", 15)
    except IOError:
        font = ImageFont.load_default()

    # Draw each point and its order
    for i, (y, x) in enumerate(points):
        # Draw the point
        draw.ellipse([(x - 5, y - 5), (x + 5, y + 5)], fill="red", outline="red")

        # Draw the order number near the point
        draw.text((x + 10, y - 10), str(i + 1), fill="blue", font=font)

    # Save or show the image
    if save_path:
        img.save(save_path)
args = get_config()

# Init the result file to store the pytorch model and other mid-result
result_path = args.result_path
os.makedirs(result_path,exist_ok=True)
print(f"the mid-result and the pytorch model will be stored in {result_path}")

# Create the model and criterion
model = get_instance(models, args.configs['model']['name'],args.configs['model'])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.load_state_dict(
    torch.load(os.path.join(args.save_dir,f"{args.split_name}_{args.configs['save_name']}")))
print("load the checkpoint in {}".format(os.path.join(args.save_dir,f"{args.split_name}_{args.configs['save_name']}")))
model.eval()
# Create the visualizations directory if it doesn't exist
config_name=os.path.basename(args.cfg).split('.')[0]
save_dir=os.path.join(args.data_path,'ridge_seg_mask')
os.makedirs(save_dir, exist_ok=True)
# os.system(f"rm -rf {save_dir}/*")
# Test the model and save visualizations
with open(os.path.join(args.data_path,'annotations.json'),'r') as f:
    data_dict=json.load(f)
img_transforms=transforms.Compose([
        # transforms.Resize((600,800)),
            transforms.ToTensor(),
            transforms.Normalize(
                # mean=[0.4623, 0.3856, 0.2822],std=[0.2527, 0.1889, 0.1334])
            mean=IMAGENET_DEFAULT_MEAN,std=IMAGENET_DEFAULT_STD)
            ])
begin=time.time()
predict=[]
labels=[]
with torch.no_grad():
    for image_name in data_dict:
        data=data_dict[image_name]
        mask=Image.open(data_dict[image_name]['mask_path']).resize((400,300),resample=Image.Resampling.NEAREST)
        mask=np.array(mask)
        mask[mask>0]=1
    
        data=data_dict[image_name]
        img = Image.open(data['enhanced_path'])
        img_tensor = img_transforms(img)
        
        img=img_tensor.unsqueeze(0).to(device)
        output_img = model(img).squeeze().cpu()
        # Resize the output to the original image size
        
        output_img=torch.sigmoid(output_img)
        output_img=output_img*mask
        seg_img=np.array(output_img*255,dtype=np.uint8)
        seg_img=Image.fromarray(seg_img)
        seg_img.save(os.path.join(save_dir,image_name))
        maxval,pred_point=k_max_values_and_indices(output_img,args.ridge_seg_number,r=60)
        value_list=[]
        point_list=[]
        for value in maxval:
            value=round(float(value),2)
            value_list.append(value)
        for x,y in pred_point:
            point_list.append([int(x),int(y)])
        data_dict[image_name]['ridge_seg']={
            "ridge_seg_path":os.path.join(save_dir,image_name),
            "value_list":value_list,
            "point_list":point_list,
            "orignal_weight":1600,
            "orignal_height":1200
        }
with open(os.path.join(args.data_path,'annotations.json'),'w') as f:
    json.dump(data_dict,f)