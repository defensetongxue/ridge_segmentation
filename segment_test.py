import json
import os
import torch
from config import get_config
from torchvision import transforms
from util import get_instance,visual_mask,k_max_values_and_indices
import models
from PIL import Image
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, roc_auc_score,recall_score
import numpy as np
# Parse arguments
import time
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
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
    torch.load(os.path.join(args.save_dir,f"clr_{args.split_name}_{args.configs['save_name']}")))
    # torch.load(os.path.join(args.save_dir,f"{args.split_name}_{args.configs['save_name']}")))
print("load the checkpoint in {}".format(os.path.join(args.save_dir,f"{args.split_name}_{args.configs['save_name']}")))
model.eval()
# Test the model and save visualizations
with open(os.path.join(args.data_path,'split',f'{args.split_name}.json'),'r') as f:
    split_list=json.load(f)['test']
with open(os.path.join(args.data_path,'annotations.json'),'r') as f:
    data_dict=json.load(f)
img_transforms=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=IMAGENET_DEFAULT_MEAN,std=IMAGENET_DEFAULT_STD
                )])
begin=time.time()
predict=[]
labels=[]
save_all_visual=True
save_all_dir=os.path.join(args.data_path,'ridge_seg')
with torch.no_grad():
    for image_name in split_list:
        mask=Image.open(data_dict[image_name]['mask_path']).resize((1600,1200),resample=Image.Resampling.BILINEAR)
        mask=np.array(mask)
        mask[mask>0]=1
    
        data=data_dict[image_name]
        img = Image.open(data['enhanced_path'])
        img_tensor = img_transforms(img)
        
        img=img_tensor.unsqueeze(0).to(device)
        output_img = model(img).cpu()
        # Resize the output to the original image size
        
        output_img=torch.sigmoid(output_img)
        output_img=F.interpolate(output_img,(1200,1600), mode='nearest')
        
        output_img=output_img*mask
        max_val=float(torch.max(output_img))
        
        output_bin=torch.where(output_img>0.5,1,0).squeeze()
        if data['stage']>0:
            tar=1
        else:
            tar=0
        if (torch.sum(output_bin)>=1):
            pred=1
        else:
            pred=0
        if pred!=tar:
            output_img=output_img.squeeze()
            
        if max_val>=0.5:
            # Construct the file path for saving the image
            ridge_seg_path = os.path.join(save_all_dir,image_name)

            # Squeeze the tensor to remove any extra dimensions
            output_img = output_img.squeeze()

            # Convert the tensor to a PIL image
            # Assuming the tensor is in the range [0, 1)
            output_img_pil = Image.fromarray((output_img.numpy() * 255).astype('uint8'))

            # Save the image
            output_img_pil.save(ridge_seg_path)
                
            # save the ridge seg for visual and sample for stage
            maxval,pred_point=k_max_values_and_indices(output_img.squeeze(),args.ridge_seg_number,r=60,threshold=0.3)
            value_list=[]
            point_list=[]
            for value in maxval:
                value=round(float(value),2)
                value_list.append(value)
            for y,x in pred_point:
                point_list.append([int(x),int(y)])
            data_dict[image_name]['ridge_seg']={
                "ridge_seg_path":ridge_seg_path,
                "value_list":value_list,
                "point_list":point_list,
                "orignal_weight":1600,
                "orignal_height":1200,
                'max_val':max_val,
                "sample_number":args.ridge_seg_number,
                "sample_interval":60
                
            }
        else:
            data_dict[image_name]['ridge_seg']={
                'max_val':max_val
            }

        labels.append(tar)
        predict.append(pred)
acc = accuracy_score(labels, predict)
auc = roc_auc_score(labels, predict)
recall=recall_score(labels,predict)
print(f"Accuracy: {acc:.4f}")
print(f"AUC: {auc:.4f}")
print(f"Recall: {recall:.4f}")

with open(os.path.join(args.data_path,'annotations.json'),'w') as f:
    json.dump(data_dict,f)