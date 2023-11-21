import json
import os
import torch
from config import get_config
from torchvision import transforms
from util import get_instance,visual_mask,visual_points
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

visual_dir=os.path.join(args.result_path,config_name)
os.makedirs(visual_dir, exist_ok=True)
os.system(f"rm -rf {visual_dir}/*")
os.makedirs(visual_dir+'/0/', exist_ok=True)
os.makedirs(visual_dir+'/1/', exist_ok=True)
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
mask=Image.open('./mask.png').resize((800,600),resample=Image.Resampling.BILINEAR)
mask=np.array(mask)
mask[mask>0]=1
with torch.no_grad():
    for image_name in split_list:
        data=data_dict[image_name]
        img = Image.open(data['enhanced_path'])
        img_tensor = img_transforms(img)
        
        img=img_tensor.unsqueeze(0).to(device)
        output_img = model(img).squeeze().cpu()
        # Resize the output to the original image size
        
        output_img=torch.sigmoid(output_img)
        output_img=torch.where(output_img>0.5,1,0)
        output_img=output_img*mask
        if 'ridge' in data:
            tar=1
        else:
            tar=0
        if (torch.sum(output_img)>=1):
            pred=1
        else:
            pred=0
        if pred!=tar:
            if tar==0:

                visual_mask(data['image_path'],output_img,str(int(torch.sum(output_img))),save_path=os.path.join(visual_dir,'0',image_name))

                visual_points(data['image_path'],output_img,
                              save_path= os.path.join(visual_dir,'0',image_name[:-4]+'_point.jpg'))
            else:

                visual_mask(data['image_path'],output_img,str(int(torch.sum(output_img))),
                            save_path=os.path.join(visual_dir,'1',image_name))

                visual_points(data['image_path'],output_img,
                              save_path= os.path.join(visual_dir,'1',image_name[:-4]+'_point.jpg'))
        labels.append(tar)
        predict.append(pred)
acc = accuracy_score(labels, predict)
auc = roc_auc_score(labels, predict)
print(f"Accuracy: {acc:.4f}")
print(f"AUC: {auc:.4f}")