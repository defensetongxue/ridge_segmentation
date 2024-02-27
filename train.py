import torch
from torch.utils.data import DataLoader
from config import get_config
from util import get_instance, train_epoch, val_epoch,get_optimizer,losses,lr_sche
from util import ridge_segmentataion_dataset as CustomDatset
from util import ridge_finetone_val
import models
import os,time
from PIL import Image
from  util.metric import Metrics
from torchvision.transforms import ToTensor,InterpolationMode,Resize
import numpy as np
# Initialize the folder
os.makedirs("checkpoints",exist_ok=True)
os.makedirs("experiments",exist_ok=True)
torch.manual_seed(0)
np.random.seed(0)

# Parse arguments
args = get_config()

# Init the result file to store the pytorch model and other mid-result
result_path = args.result_path
os.makedirs(result_path,exist_ok=True)
print(f"using config file {args.cfg}")
print(f"the mid-result and the pytorch model will be stored in {result_path}")

# Create the model and criterion
model = get_instance(models, args.configs['model']['name'],args.configs['model'])
criterion=get_instance(losses,args.configs['model']['loss_func'],pos_weight=args.configs['model']['loss_weight'])
model.train()
# Creatr optimizer
optimizer = get_optimizer(args.configs, model)
lr_scheduler=lr_sche(config=args.configs["lr_strategy"])
last_epoch = args.configs['train']['begin_epoch']

# Load the datasets
train_dataset=CustomDatset(args.data_path,'train',split_name=args.split_name,factor=args.configs['model']['factor'])
val_dataset=ridge_finetone_val(args.data_path,split_name=args.split_name,split='val',postive_cnt=1e5)
test_dataset=ridge_finetone_val(args.data_path,split_name=args.split_name,split='test',postive_cnt=1e5)
# Create the data loaders
train_loader = DataLoader(train_dataset, 
                          batch_size=args.configs['train']['batch_size'],
                          shuffle=True, num_workers=args.configs['num_works'])
val_loader = DataLoader(val_dataset,
                        batch_size=args.configs['train']['test_bc'],
                        shuffle=False, num_workers=args.configs['num_works'])
test_loader = DataLoader(test_dataset,
                        batch_size=args.configs['train']['test_bc'],
                        shuffle=False, num_workers=args.configs['num_works'])
metric=Metrics("Main")
print("There is {} patch size".format(args.configs['train']['batch_size']))
print(f"Train: {len(train_loader)}, Val: {len(val_loader)}")
# Set up the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
print(f"using {device} for training")
# Set up the optimizer, loss function, and early stopping

early_stop_counter = 0
best_val_loss = float('inf')
total_epoches = args.configs['train']['end_epoch']
max_auc=0
max_recall=0
save_epoch=-1
mask=Image.open('./mask.png').convert('L')
mask =Resize((int(1200*args.configs['model']['factor']),int(1600*args.configs['model']['factor'])),interpolation=InterpolationMode.NEAREST)(mask)
mask=ToTensor()(mask)
mask[mask>0]=1
mask=mask.unsqueeze(0)
# Training and validation loop
for epoch in range(last_epoch, total_epoches):
    
    start_time = time.time()  # Record the start time of the epoch
    train_loss = train_epoch(model, optimizer, train_loader, criterion, device,lr_scheduler,epoch)
    val_loss,metric = val_epoch(model, val_loader, criterion, device,metric,mask)
    
    end_time = time.time()  # Record the end time of the epoch
    elapsed_time = end_time - start_time  # Calculate the elapsed time
    elapsed_hours = elapsed_time / 3600  # Convert elapsed time to hours
    print(f"Epoch {epoch + 1}/{total_epoches}, "
          f"Train Loss: {train_loss:.6f}, "
          f"Val Loss: {val_loss:.6f} "
          f"Time: {elapsed_hours:.2f} hours")
    
    print(metric)
    # Update the learning rate if using ReduceLROnPlateau or CosineAnnealingLR
    if metric.image_auc > max_auc:
        save_epoch=epoch
        max_auc=metric.image_auc
        early_stop_counter = 0
        torch.save(model.state_dict(),
                   os.path.join(args.save_dir,f"{args.split_name}_{args.configs['save_name']}"))
        print("Model saved as {}".format(os.path.join(args.save_dir,f"{args.split_name}_{args.configs['save_name']}")))
    else:
        early_stop_counter+=1
        if early_stop_counter>args.configs['train']["early_stop"]:
            break
    metric.reset()
# model.load_state_dict(
#     torch.load(os.path.join(args.save_dir,f"{args.split_name}_{args.configs['save_name']}"))
# # )
# test_loss,metric = val_epoch(model, test_loader, criterion, device,metric,mask)
# print(metric)
# key=f'{str(os.path.basename(args.cfg)[:-5])}'
# metric._store(key,args.split_name,save_epoch)