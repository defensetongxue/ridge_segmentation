import json,os
with open('./record.json','r') as f:
    orignal_record=json.load(f)
res={}
for model_name in orignal_record:
    model_res={
        "accuracy": 0.0,
            "auc": 0.0,
            "recall": 0.0
    }
    for split in orignal_record[model_name]:
        model_res['accuracy']+=orignal_record[model_name][split]['accuracy']
        model_res['auc']+=orignal_record[model_name][split]['auc']
        model_res['recall']+=orignal_record[model_name][split]['recall']
    model_res['accuracy']=round(model_res['accuracy']/4,4)
    model_res['auc']=round(model_res['auc']/4,4)
    model_res['recall']=round(model_res['recall']/4,4)
    res[model_name]=model_res
with open('./paser_record.json','w') as f:
    json.dump(res,f)