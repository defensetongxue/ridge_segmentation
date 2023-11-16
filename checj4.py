import os,json
from os.path import join
from shutil import copy
from PIL import Image,ImageDraw,ImageFont
import numpy as np
data_path='../autodl-tmp/dataset_ROP'
with open(join(data_path,'annotations.json'),'r') as f:
    data_dict=json.load(f)
with open('./4.json','r') as f:
    ridge_extra=json.load(f)
os.makedirs('./experiments/check',exist_ok=True)
for image_name in ridge_extra:
    if 'ridge_diffusion_path' in data_dict[image_name]:
        copy(data_dict[image_name]['ridge_diffusion_path'],'./experiments/check/'+image_name[:-4]+'_rid.jpg')

