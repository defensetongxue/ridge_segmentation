import json
import os
import torch
from config import get_config
from torchvision import transforms
from utils_ import recompone_overlap,get_instance,visual_mask
import models
from PIL import Image
# Parse arguments
TEST_CNT=10
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
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4623, 0.3856, 0.2822],
                std=[0.2527, 0.1889, 0.1334])])
with torch.no_grad():
    for data in test_data:
        img_path=data['image_path']
        img_name=data['image_name']
        img=Image.open(img_path)
        img=img_transforms(img)

        patch_size = args.patch_size
        stride = patch_size
        patches = img.unfold(2, patch_size, stride).unfold(3, patch_size, stride)
        patches = patches.permute(0, 2, 3, 1, 4, 5).reshape(-1, 3, patch_size, patch_size)
        output_patches = model(patches.to(device))

        # Resize the output to the original image size
        mask = recompone_overlap(output_patches.cpu().numpy(), img.shape[2], img.shape[3], stride, stride)
        visual_mask(img_path,mask,os.path.join(result_path,img_name))
print("Finished testing")