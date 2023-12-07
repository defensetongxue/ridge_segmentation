import os,json
from configs import get_config
if __name__ =='__main__':
    args=get_config()
    data_path=args.data_path
    split_name=args.split_name
    
    with open(os.path.join(data_path,'annotations.json'),'r') as f:
        data_dict=json.load(f)
    with open(os.path.join(data_path,'split',f'{split_name}.json'),'r')  as f:
        split_list= json.load(f)
    new={'train':[],'val':[],'test':[]}
    for split in split_list:
        for image_name in split_list[split]:
            if not data_dict[image_name]['suspicious']:
                new[split].append(image_name)
                assert data_dict[image_name]['stage']==0
            elif data_dict[image_name]['stage']>0:
                new[split].append(image_name)
    with open(os.path.join(data_path,'split',f'clr_{split_name}.json'),'w') as f:
        json.dump(new,f)
        
    