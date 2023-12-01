import json
import numpy as np
import torchvision.transforms as transforms
import torch
from PIL import Image
from scipy.ndimage import zoom
import os
from scipy.ndimage import gaussian_filter
from .api_record import api_check,api_update
def generate_ridge_diffusion(data_path):
    print("begin to generate ridge annotations")
    api_check(data_path,'ridge')
    os.makedirs(os.path.join(data_path,'ridge_diffusion'),exist_ok=True)
    os.system(f"rm -rf {os.path.join(data_path,'ridge_diffusion')}/*")
    with open(os.path.join(data_path,'annotations.json'),'r') as f:
        data_list=json.load(f)
    for image_name in data_list:
        data=data_list[image_name]
        if not 'ridge' in data:
            continue
        mask = generate_diffusion_heatmap(data['image_path']
                                          ,data['ridge']['ridge_coordinate'], 
                                          factor=0.5,
                                          Gauss=False)
        mask_save_name=data['id']+'.png'
        mask_save_path=os.path.join(data_path,'ridge_diffusion',mask_save_name)
        Image.fromarray((mask * 255).astype(np.uint8)).save(mask_save_path)
        data_list[image_name]['ridge_diffusion_path']=mask_save_path
    with open(os.path.join(data_path,'annotations.json'),'w') as f:
        json.dump(data_list,f)
    api_update(data_path,'ridge_diffusion_path','path to ridge diffusion mask')
    print("finish")
def generate_diffusion_heatmap(image_path,points,factor=0.5,Gauss=False):
    '''
    factor is the compress rate: for stability, we firt apply a conv layer to 
    downsample the original data
    '''
    img_tensor=imge_tensor_compress(image_path,factor)

    img = Image.open(image_path).convert('RGB')
    img=transforms.ToTensor()(img)
    heatmap_width=int(img.shape[1]*factor)
    heatmap_height=int(img.shape[2]*factor)
    heatmap = np.zeros((heatmap_width, heatmap_height), dtype=np.float32)
    new_points=[]
    for x,y in points:
        new_points.append([int(x*factor),int(y*factor)])
    points=np.array(new_points)

    heatmap = generate_heatmap(heatmap,img_tensor,points,image_path)
    if Gauss:
        # as we are using bce loss as loss function, no 
        # gauss map is needed
        Gauss_heatmap=gaussian_filter(heatmap,3)
        heatmap=np.where(heatmap>Gauss_heatmap,heatmap,Gauss_heatmap)
    mask=heatmap2mask(heatmap,int(1/factor))
    return mask

def heatmap2mask(heatmap,factor=4):
    mask = zoom(heatmap, factor)
    return mask

def norm_tensor(t):
    min_v=t.min()
    r_val=t.max()-min_v
    return (t-min_v)/r_val

def visual_mask(image_path, mask,save_path='./tmp.jpg'):
    # Open the image file.
    image = Image.open(image_path).convert("RGBA")  # Convert image to RGBA

    # Create a blue mask.
    mask_np = np.array(mask)
    mask_blue = np.zeros((mask_np.shape[0], mask_np.shape[1], 4), dtype=np.uint8)  # 4 for RGBA
    mask_blue[..., 2] = 255  # Set blue channel to maximum
    mask_blue[..., 3] = (mask_np * 127.5).astype(np.uint8)  # Adjust alpha channel according to the mask value

    # Convert mask to an image.
    mask_image = Image.fromarray(mask_blue)

    # Overlay the mask onto the original image.
    composite = Image.alpha_composite(image, mask_image)

    # Convert back to RGB mode (no transparency).
    rgb_image = composite.convert("RGB")

    # Save the image with mask to the specified path.
    rgb_image.save(save_path)

def imge_tensor_compress(img_path,factor):
    img=Image.open(img_path)
    img=transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor(),
    ])(img).unsqueeze(0)

    # Calculate patch size based on the downscale factor
    patch_size = int(1/factor)
    
    # Create averaging kernel
    kernel = torch.full((1, img.shape[1], patch_size, patch_size), 1./(patch_size * patch_size))
    
    # Apply convolution operation to get patch embedding
    img =torch.nn.functional.conv2d(img, kernel, stride=patch_size)
    img=img.squeeze()
    return img

def generate_heatmap(heatmap,img_tensor, points,image_path):
    points=generate_point_sequence(points)
    for i in range(points.shape[0]-1):
        heatmap=diffusion(heatmap,img_tensor,points[i],points[i+1],image_path)

    return heatmap

def generate_point_sequence(points):
    """
    Generate a sequence of points ordered based on proximity, starting from
    the point with the smallest x or y value.
    
    points: np.array, shape (n_points, 2)
    
    Returns: np.array, shape (n_points, 2)
    """
    # Decide axis
    x_range = np.ptp(points[:, 0])  # Peak to peak (max - min) for x
    y_range = np.ptp(points[:, 1])  # Peak to peak (max - min) for y
    
    # Sort by x or y depending on range
    if x_range > y_range:
        axis = 0  # x-axis
    else:
        axis = 1  # y-axis
    
    # Find the starting point (x0 or y0)
    start_idx = np.argmin(points[:, axis])
    start_point = points[start_idx]
    
    # Initialize sequence with the starting point
    sequence = [start_point]
    
    # Initialize set of unvisited points
    unvisited = set(range(len(points)))
    unvisited.remove(start_idx)
    
    # Greedy algorithm to find the closest points
    current_point = start_point
    while unvisited:
        # Find the closest point to the current point
        next_idx = min(unvisited, key=lambda idx: get_distance(current_point, points[idx]))
        next_point = points[next_idx]
        
        # Update current point and sequence
        current_point = next_point
        sequence.append(current_point)
        
        # Remove the next point from the unvisited set
        unvisited.remove(next_idx)
    
    return np.array(sequence)
def get_distance(p0, p1):
    return ((p0[0]-p1[0])**2 + (p0[1]-p1[1])**2)
def get_similarity(img_tensor,p0,p1):
    return 1-((img_tensor[p0[1],p0[0]]-img_tensor[p1[1],p1[0]])**2)


def diffusion(heatmap,img_tensor,p0,p1,image_path):
    heatmap[p0[1],p0[0]]=1
    heatmap[p1[1],p1[0]]=1
    px=p0
    now_distance=get_distance(px,p1)
    while(now_distance>1):
        max_simi=-9e5
        rem=[]
        for i in [-1,0,1]:
            for j in [-1,0,1]:
                p=[px[0]+i,px[1]+j]
                p[0]=max(0,min(799,p[0]))
                p[1]=max(0,min(599,p[1]))
                if heatmap[p[1],p[0]]>0:
                    continue
                if get_distance(p,p1)>now_distance:
                    continue
                simi=get_similarity(img_tensor,p,p0)+get_similarity(img_tensor,p,p1)

                if simi>max_simi:
                    max_simi=simi
                    rem=[i,j]
        if len(rem)==0:
            print(image_path)
            return heatmap
        px=[px[0]+rem[0],px[1]+rem[1]]
        now_distance=get_distance(px,p1)
        heatmap[px[1],px[0]]=1
    return heatmap