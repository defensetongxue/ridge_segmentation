import json
import os
import torch
from config import get_config
from torchvision import transforms
from utils_ import get_instance,visual_mask,patchtify
import models
from PIL import Image
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
# Test the model and save visualizations
with open(os.path.join(data_path,'ridge','test.json'),'r') as f:
    test_data=json.load(f)[:TEST_CNT]
img_transforms=transforms.Compose([
            transforms.Resize((600,800)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4623, 0.3856, 0.2822],
                std=[0.2527, 0.1889, 0.1334])])
begin=time.time()
with torch.no_grad():
    for data in test_data:
        img_path=data['image_path']
        img_name=data['image_name']
        img=Image.open(img_path)
        img=img_transforms(img).unsqueeze(0)

        output_img = model(img.to(device)).cpu()
        # Resize the output to the original image size
        
        mask=torch.sigmoid(output_img).numpy()
        mask=zoom(mask,2)
        visual_mask(img_path,mask,os.path.join(result_path,img_name))
end=time.time()
print(f"Finished testing. Time cost {(end-begin)/100:.4f}")