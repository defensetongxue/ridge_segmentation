import json
import os
import torch
from config import get_config
from torchvision import transforms
from util import get_instance,visual_mask,visual_points
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
    # transforms.Resize((600,800)),
            transforms.ToTensor(),
            transforms.Normalize(
                # mean=[0.4623, 0.3856, 0.2822], std=[0.2527, 0.1889, 0.1334]
                mean=IMAGENET_DEFAULT_MEAN,std=IMAGENET_DEFAULT_STD
                )])
begin=time.time()
predict=[]
labels=[]

val_list_postive=[]
val_list_negtive=[]
val_list=[]
with torch.no_grad():
    for image_name in split_list:
        mask=Image.open(data_dict[image_name]['mask_path']).resize((800,600),resample=Image.Resampling.BILINEAR)
        mask=np.array(mask)
        mask[mask>0]=1
    
        data=data_dict[image_name]
        img = Image.open(data['enhanced_path'])
        img_tensor = img_transforms(img)
        
        img=img_tensor.unsqueeze(0).to(device)
        output_img = model(img).cpu()
        # Resize the output to the original image size
        
        output_img=torch.sigmoid(output_img)
        max_val=float(torch.max(output_img))
        val_list.append(max_val)
        output_img=F.interpolate(output_img,(600,800), mode='nearest')
        output_img=output_img*mask
        
        output_bin=torch.where(output_img>0.5,1,0).squeeze()
        if data['stage']>0:
            tar=1
            val_list_postive.append(max_val)
        else:
            tar=0
            val_list_negtive.append(max_val)
        if (torch.sum(output_bin)>=1):
            pred=1
        else:
            pred=0
        if pred!=tar:
            output_img=output_img.squeeze()
            if tar==0:

                visual_mask(data['image_path'],output_img,str(round(max_val,2)),save_path=os.path.join(visual_dir,'0',image_name))

                # visual_points(data['image_path'],output_img,
                #               save_path= os.path.join(visual_dir,'0',image_name[:-4]+'_point.jpg'))
            else:
                # gt=Image.open(data['ridge_diffusion_path']).convert('L')
                # gt=transforms.Resize((600,800))(gt)
                # gt=np.array(gt)
                # gt[gt>0]=1
                # visual_mask(data['image_path'],gt,str(int(torch.sum(output_img))),
                #             save_path=os.path.join(visual_dir,'1',image_name[:-4]+'_point.jpg'))
                visual_mask(data['image_path'],output_img,str(round(max_val,2)),
                            save_path=os.path.join(visual_dir,'1',image_name))

                # visual_points(data['image_path'],output_img,
                #               save_path= os.path.join(visual_dir,'1',image_name[:-4]+'_point.jpg'))
        labels.append(tar)
        predict.append(pred)

acc = accuracy_score(labels, predict)
auc = roc_auc_score(labels, predict)
recall=recall_score(labels,predict)
print(f"Accuracy: {acc:.4f}")
print(f"AUC: {auc:.4f}")
print(f"Recall: {recall:.4f}")
# import matplotlib.pyplot as plt
# # Bin data with width 0.2
# bins = np.arange(0, 1.01, 0.05)  # Bins from 0 to 1 with step size 0.2
# positive_hist, _ = np.histogram(val_list_postive, bins=bins, density=True)
# negative_hist, _ = np.histogram(val_list_negtive, bins=bins, density=True)

# # Calculate proportions for each bin
# positive_hist *= 0.05  # width of bins
# negative_hist *= 0.05

# # Plot with overlapping bars
# plt.bar(bins[:-1], positive_hist, width=0.05, align='center', alpha=0.5, color='blue', label='Positive')
# plt.bar(bins[:-1], negative_hist, width=0.05, align='center', alpha=0.5, color='orange', label='Negative')
# plt.xlabel('Value Range')
# plt.ylabel('Proportion')
# plt.title('Value Distribution')
# plt.xticks(bins)
# plt.legend()

# # Save the figure
# plt.savefig('./save.png')
# val_list=np.array(val_list)
# acc_list=[]
# auc_list=[]
# recall_list=[]

# for judge_val in np.arange(0.3, 0.51, 0.01):
#     pred_label=val_list>judge_val
#     acc_list.append(accuracy_score(labels,pred_label))
#     auc_list.append(roc_auc_score(labels,pred_label))
#     recall_list.append(recall_score(labels,pred_label))
    
# # Create range for x-axis
# x_range = np.arange(0.3, 0.51, 0.01)

# # Plotting
# plt.figure(figsize=(10, 6))
# plt.plot(x_range, acc_list, label='Accuracy', color='blue')
# plt.plot(x_range, auc_list, label='AUC', color='green')
# plt.plot(x_range, recall_list, label='Recall', color='red')

# plt.xlabel('Judgement Value')
# plt.ylabel('Score')
# plt.title('Metric Scores at Different Judgement Values')
# plt.legend()
# plt.grid(True)
# plt.savefig('./save_line.png')

