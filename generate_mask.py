import os,json
from PIL import Image
import numpy as np
data_path='../autodl-tmp/dataset_ROP'
with open(os.path.join(data_path,'annotations.json'),'r') as f:
    data_dict=json.load(f)
os.makedirs(os.path.join(data_path,'mask'))

def generate_mask(image_path,save_path):
    img=Image.open(image_path)
    img=np.array(img)/255
    img=np.sum(img,axis=-1)
    mask=np.where(img<0.26,0,1)
    mask_img = Image.fromarray(np.uint8(mask * 255), 'L')
    mask_img.save(save_path)
for image_name in data_dict:
    save_mask_path=os.path.join(data_path,'mask',image_name)
    generate_mask(data_dict[image_name]['image_path'],
                  save_mask_path)
    data_dict[image_name]['mask_path']=save_mask_path
with open(os.path.join(data_path,'annotations.json'),'w') as f:
    json.dump(data_dict,f)