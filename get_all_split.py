import os,json
data_path='../autodl-tmp/dataset_ROP'
with open(os.path.join(data_path,'annotations.json')) as f:
    data_dict=json.load(f)
with open(os.path.join(data_path,'split','clr_1.json')) as f:
    split_list=json.load(f)
all_split={'train':[],'val':[],'test':[]}
all_split['train']=split_list['train']+split_list['val']
all_split['val']=split_list['test']
def p_info(a):
    for k in a:
        print(len(a[k]))
p_info(split_list)
p_info(all_split)
with open(os.path.join(data_path,'split','all.json'),'w') as f:
    json.dump(all_split,f)