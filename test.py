import json
import os
import torch
from config import get_config
from torchvision import transforms
from utils_ import get_instance,visual_mask,visual_points
import models
from PIL import Image
import torch.nn.functional as F
from scipy.ndimage import zoom
# Parse arguments
TEST_CNT=100
import time
args = get_config()

# Init the result file to store the pytorch model and other mid-result
data_path=args.path_tar
result_path = args.result_path
os.makedirs(result_path,exist_ok=True)
print(f"the mid-result and the pytorch model will be stored in {result_path}")

# Create the model and criterion
model = get_instance(models, args.model)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.load_state_dict(
    torch.load(args.save_name))
print(f"load the checkpoint in {args.save_name}")
model.eval()
# Create the visualizations directory if it doesn't exist

visual_dir = os.path.join(args.result_path, 'visual')
os.makedirs(visual_dir, exist_ok=True)
os.makedirs(os.path.join(args.result_path,'visual_points'),exist_ok=True)
# Test the model and save visualizations
with open(os.path.join(data_path,'ridge','test.json'),'r') as f:
    data_list=json.load(f)[:TEST_CNT]
img_transforms=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4623, 0.3856, 0.2822],
                std=[0.2527, 0.1889, 0.1334])])
begin=time.time()
with torch.no_grad():
    for data in data_list:
        img = Image.open(data['image_path']).convert('RGB')
        img_tensor = img_transforms(img)
        pos_path=os.path.join(data_path,'pos_embed',data['image_name'].split('.')[0]+'.pt')
        pos_embed=torch.load(pos_path)
        pos_embed=F.interpolate(pos_embed[None,None,:,:], size=img_tensor.shape[-2:], mode='nearest')
        pos_embed=pos_embed.squeeze()
        
        img=img_tensor.unsqueeze(0).to(device)
        pos_embed=pos_embed.unsqueeze(0).to(device)
        output_img = model((img,pos_embed)).cpu()
        # Resize the output to the original image size
        
        mask=torch.sigmoid(output_img).numpy()
        visual_mask(data['image_path'],mask,os.path.join(visual_dir,data['image_name']))
        if True:
            visual_points(data['image_path'],mask,
                          save_path= os.path.join(args.result_path,'visual_points',data['image_name']))
end=time.time()
print(f"Finished testing. Time cost {(end-begin)/100:.4f}")
