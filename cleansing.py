import json
import os
import torch
from utils_ import generate_diffusion_heatmap
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

def generate_segmentation_mask(data_path, patch_size, stride):
    # Generate path_image folder
    # os.makedirs(os.path.join(data_path,'ridge_seg','images'),exist_ok=True)
    os.makedirs(os.path.join(data_path,'ridge_seg','masks'),exist_ok=True)
    os.makedirs(os.path.join(data_path,'ridge_seg','annotations'),exist_ok=True)
    os.system('rm -rf ')
    splits=['train','val']
    for split in splits:
        with open(os.path.join(data_path,'ridge',f'{split}.json'),'r') as f:
            ridge_data_list=json.load(f)
        ridge_map={ridge_data_list[i]['image_name']:i for i in range(len(ridge_data_list))}
        annotate=[]
        with open(os.path.join(data_path,'annotations',f'{split}.json'),'r') as f:
            data_list =json.load(f)
        for  data in data_list:
            if data['image_name'] in ridge_map.keys():
                id_cnt=ridge_map[data['image_name']]
                mask = generate_diffusion_heatmap(data['image_path'],ridge_data_list[id_cnt]['ridge_coordinate'], factor=0.5, Gauss=False)
            else:
                image=Image.open(data['image_path'])
                h,w=image.size
                mask=np.zeros((w,h))
            # Load image and mask
            image_id=data['image_name'].split('.')[0]
            # Save mask patch
            mask_path = os.path.join(data_path, 'ridge_seg', 'masks', f"{image_id}.png")
            Image.fromarray((mask * 255).astype(np.uint8)).save(mask_path)
            # Get class
            class_annote = data["class"] 
            annotate.append({
                        "img_path": data['image_path'],
                        "mask_path": mask_path,
                        "class": class_annote
                    })

        with open(os.path.join(data_path, 'ridge_seg', 'annotations', f'{split}.json'), 'w') as f:
            json.dump(annotate, f)



def parse_json(input_data,label_class=0,image_dict="../autodl-tmp/images"):
    annotations = input_data.get("annotations", [])
    if annotations:
        result = annotations[0].get("result", [])
    image_name=input_data["file_upload"].split('-')[-1]
    new_data = {
        "image_name": image_name,
        "image_path":os.path.join(image_dict,image_name),
        "ridge_number": 0,
        "ridge_coordinate": [],
        "other_number": 0,
        "other_coordinate": [],
        "plus_number": 0,
        "plus_coordinate": [],
        "pre_plus_number": 0,
        "pre_plus_coordinate": [],
        "class": label_class
    }

    for item in result:
        if item["type"] == "keypointlabels":
            # x, y = item["value"]["x"], item["value"]["y"]
            x= item["value"]["x"]*item["original_width"]/100
            y= item["value"]["y"]*item["original_height"]/100
            label = item["value"]["keypointlabels"][0]

            if label == "Ji":
                new_data["ridge_number"] += 1
                new_data["ridge_coordinate"].append((x, y))
            elif label == "Other":
                new_data["other_number"] += 1
                new_data["other_coordinate"].append((x, y))
            elif label == "Plus":
                new_data["plus_number"] += 1
                new_data["plus_coordinate"].append((x, y))
            elif label == "Pre-plus":
                new_data["pre_plus_number"] += 1
                new_data["pre_plus_coordinate"].append((x, y))

    return new_data

def parse_json_file(file_dict,data_path):
    
    annotation=[]
    file_list=sorted(os.listdir(file_dict))
    print(f"read the origianl json file from {file_list}")
    for file in file_list:
        if not file.split('.')[-1]=='json':
            print(f"unexpected file {file} in json_src")
            continue
        if not file.split('.')[-1]=='json':
            print(f"unexpected file {file} in json_src")
            continue
        with open(os.path.join(file_dict,file), 'r') as f:
            data = json.load(f)
        
        for json_obj in data:
            new_data=parse_json(json_obj,label_class=int(file[0]),
                                image_dict=os.path.join(data_path,'images'))
            if new_data["ridge_number"]>0:        
                annotation.append(new_data)

    return annotation

def split_data(data_path, annotations):
    # Important: do not shuffle, as there may be some images very similar be split into different set
    os.makedirs(os.path.join(data_path, 'ridge'),exist_ok=True)
    os.system(f"rm -rf {os.path.join(data_path, 'ridge')}/*")

    with open(os.path.join(data_path, 'annotations', "train.json"), 'r') as f:
        train_list=json.load(f)
        train_list=[i['image_name'] for i in train_list]
    with open(os.path.join(data_path, 'annotations', "val.json"), 'r') as f:
        val_list=json.load(f)
        val_list=[i['image_name'] for i in val_list]
    with open(os.path.join(data_path, 'annotations', "test.json"), 'r') as f:
        test_list=json.load(f)
        test_list=[i['image_name'] for i in test_list]

    train_annotations = []
    val_annotations = []
    test_annotations =[]
    train_condition={'1':0,"2":0,"3":0}
    val_condition={'1':0,"2":0,"3":0}
    test_condition={'1':0,"2":0,"3":0}
    for data in annotations:
        if data['image_name'] in train_list:
            train_annotations.append(data)
            train_condition[str(data['class'])]+=1
        if data['image_name'] in val_list:
            val_annotations.append(data)
            val_condition[str(data['class'])]+=1
        if data['image_name'] in test_list:
            test_annotations.append(data)
            test_condition[str(data['class'])]+=1
    with open(os.path.join(data_path, 'ridge', 'train.json'), 'w') as f:
        json.dump(train_annotations, f, indent=2)

    with open(os.path.join(data_path, 'ridge', 'val.json'), 'w') as f:
        json.dump(val_annotations, f, indent=2)

    with open(os.path.join(data_path, 'ridge', 'test.json'), 'w') as f:
        json.dump(test_annotations, f, indent=2)

    print(f"Total samples: {len(annotations)}"  )
    print(f"Train samples: {len(train_annotations)} {train_condition} {train_condition}")
    print(f"Validation samples: {len(val_annotations)} {val_condition} {val_condition}")
    print(f"Test samples: {len(test_annotations)} {test_condition} {test_condition}")
if __name__=='__main__':
    from config import get_config
    args=get_config()
    
    # cleansing
    if args.generate_ridge:
        
        annotations=parse_json_file(args.json_file_dict,args.path_tar)
        split_data(args.path_tar,annotations)
        print(f"generate ridge_coordinate in {os.path.join(args.path_tar,'ridge')}")
    generate_segmentation_mask(args.path_tar,args.patch_size,args.stride)