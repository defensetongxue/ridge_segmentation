import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, roc_auc_score,recall_score
from config import get_config
from PIL import Image
from models import get_transUnet
import os,time,json
import numpy as np
from torchvision import transforms
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
args = get_config()
# Init the result file to store the pytorch model and other mid-result
result_path = args.result_path
os.makedirs(result_path,exist_ok=True)
print(f"the mid-result and the pytorch model will be stored in {result_path}")

# Create the model and criterion
model = get_transUnet(512,1)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.load_state_dict(
    torch.load(os.path.join(args.save_dir,f"{args.split_name}_{args.configs['save_name']}")))
print("load the checkpoint in {}".format(os.path.join(args.save_dir,f"{args.split_name}_{args.configs['save_name']}")))
model.eval()
# Test the model and save visualizations
with open(os.path.join(args.data_path,'split',f'{args.split_name}.json'),'r') as f:
    split_list=json.load(f)['test']
with open(os.path.join(args.data_path,'annotations.json'),'r') as f:
    data_dict=json.load(f)

img_transforms=transforms.Compose([
            transforms.Resize((512,512)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=IMAGENET_DEFAULT_MEAN,std=IMAGENET_DEFAULT_STD
                )])
begin=time.time()
predict=[]
labels=[]
val_list_postive=[]
val_list_negtive=[]
val_list=[]

visual_error=False
if visual_error:
    config_name=os.path.basename(args.cfg).split('.')[0]
    visual_dir=os.path.join(args.result_path,config_name)
    os.makedirs(visual_dir, exist_ok=True)
    os.makedirs(visual_dir+'/0/', exist_ok=True)
    os.makedirs(visual_dir+'/1/', exist_ok=True)

with torch.no_grad():
    for image_name in split_list:
        mask=Image.open(data_dict[image_name]['mask_path']).resize((512,512),resample=Image.Resampling.BILINEAR)
        mask=np.array(mask)
        mask[mask>0]=1
    
        data=data_dict[image_name]
        img = Image.open(data['enhanced_path'])
        img_tensor = img_transforms(img)
        
        img=img_tensor.unsqueeze(0).to(device)
        output_img = model(img).cpu()
        # Resize the output to the original image size
        
        output_img=torch.sigmoid(output_img)
        output_img=F.interpolate(output_img,(512,512), mode='nearest')
        
        output_img=output_img*mask
        max_val=float(torch.max(output_img))
        val_list.append(max_val)
        
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
            if visual_error:
                raise# not implement
        
        labels.append(tar)
        predict.append(pred)
acc = accuracy_score(labels, predict)
auc = roc_auc_score(labels, predict)
recall=recall_score(labels,predict)
print(f"Accuracy: {acc:.4f}")
print(f"AUC: {auc:.4f}")
print(f"Recall: {recall:.4f}")
# Check if the record file exists and load it; if not, initialize an empty dict


record_path = './experiments/record.json'
if os.path.exists(record_path):
    with open(record_path, 'r') as f:
        record = json.load(f)
else:
    record = {}

# Update the record for the current model and split
args = get_config()  # Make sure this returns the correct configuration
if 'transunet' not in record:
    record['transunet'] = {}
parameter_key=f"{str(args.lr)}_{str(args.wd)}"
if parameter_key not in record['transunet']:
    record['transunet'][parameter_key]={}
# Correct the syntax for storing metrics in the dictionary
record['transunet'][parameter_key][args.split_name] = {
    "Accuracy": f"{acc:.4f}",
    "AUC": f"{auc:.4f}",
    "Recall": f"{recall:.4f}"
}

# Write the updated record back to the file
with open(record_path, 'w') as f:
    json.dump(record, f, indent=4)