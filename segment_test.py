import json
import os
import torch
from config import get_config
from torchvision import transforms
from utils_ import get_instance,k_max_values_and_indices
import models
from PIL import Image
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, roc_auc_score
import numpy as np
# Parse arguments
import time
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
os.system(f"rm -rf {save_dir}/*")
# Test the model and save visualizations
with open(os.path.join(args.data_path,'split',f'{args.split_name}.json'),'r') as f:
    split_list=json.load(f)['test']
with open(os.path.join(args.data_path,'annotations.json'),'r') as f:
    data_dict=json.load(f)
img_transforms=transforms.Compose([
        transforms.Resize((600,800)),
    
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4623, 0.3856, 0.2822],
                std=[0.2527, 0.1889, 0.1334])])
begin=time.time()
predict=[]
labels=[]
mask=Image.open('./mask.png')
mask=np.array(mask)
mask[mask>0]=1
with torch.no_grad():
    for image_name in split_list:
        data=data_dict[image_name]
        img = Image.open(data['enhanced_path']).convert('RGB')
        img_tensor = img_transforms(img)
        
        img=img_tensor.unsqueeze(0).to(device)
        output_img = model(img).squeeze().cpu()
        # Resize the output to the original image size
        
        output_img=torch.sigmoid(output_img)
        output_img=np.array(output_img*255,dtype=np.uint8)
        seg_img=Image.fromarray(output_img)
        seg_img.save(os.path.join(save_dir,image_name))
        maxval,pred_point=k_max_values_and_indices(output_img,args.ridge_seg_number)
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
            "point_list":point_list
        }
with open(os.path.join(args.data_path,'annotations.json'),'w') as f:
    json.dump(data_dict,f)