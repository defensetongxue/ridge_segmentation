import json
import os
import torch
from config import get_config
from torchvision import transforms
from util import get_instance,ridge_enhance,visual_mask
import models
from PIL import Image
import numpy as np
# Parse arguments
import time
args = get_config()
data_path=args.data_path
# Init the result file to store the pytorch model and other mid-result

# Create the model and criterion
model = get_instance(models, args.configs['model']['name'],args.configs['model'])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.load_state_dict(
    torch.load(os.path.join(args.save_dir,f"{args.split_name}_{args.configs['save_name']}")))
print("load the checkpoint in {}".format(os.path.join(args.save_dir,f"{args.split_name}_{args.configs['save_name']}")))
model.eval()
# Create the visualizations directory if it doesn't exist
visual_dir=os.path.join(data_path,'ridge_enhance')

os.makedirs(visual_dir, exist_ok=True)
# Test the model and save visualizations
with open(os.path.join(args.data_path,'split',f'{args.split_name}.json'),'r') as f:
    split_list=json.load(f)
with open(os.path.join(args.data_path,'annotations.json'),'r') as f:
    data_dict=json.load(f)
map_split={}
for split in split_list:
    for image_name in split_list[split]:
        map_split[image_name]=split
img_transforms=transforms.Compose([
    transforms.Resize((600,800)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4623, 0.3856, 0.2822],
                std=[0.2527, 0.1889, 0.1334])])
begin=time.time()
predict=[]
labels=[]
mask=Image.open('./mask.png').resize((800,600),resample=Image.Resampling.BILINEAR)
mask=np.array(mask)
mask[mask>0]=1
new_split={'train':[],'val':[],'test':[],"train_norm":[],"val_norm":[],"test_norm":[]}
with torch.no_grad():
    for image_name in data_dict:
        data=data_dict[image_name]
        img = Image.open(data['enhanced_path'])
        img_tensor = img_transforms(img)
        
        img=img_tensor.unsqueeze(0).to(device)
        output_img = model(img).squeeze().cpu()
        # Resize the output to the original image size
        
        output_img=torch.sigmoid(output_img)
        
        if torch.max(output_img)>0.4:
            new_split[map_split[image_name]].append(image_name)
            if data['stage']==0:
               visual_mask(image_path=data['image_path'],
                            mask=output_img,
                    save_path='./experiments/checkUnsuc/'+image_name) 
print(len(os.listdir('./experiments/checkUnsuc/')))

        # else:
#             if data['stage']>0:
#                 print(image_name,' ',map_split[image_name])
#                 visual_mask(image_path=data['image_path'],
#                             mask=output_img,
#                     save_path='./experiments/checkUnsuc/'+image_name)
#             new_split[map_split[image_name]+"_norm"].append(image_name)
# with open(os.path.join(args.data_path,'split',f'ri_{args.split_name}.json'),'w') as f:
#     json.dump(new_split,f)