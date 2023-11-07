import torch
from torch.utils.data import DataLoader
from config import get_config
from utils_ import get_instance, train_epoch, fineone_val_epoch,get_optimizer,losses,lr_sche
from utils_ import ridge_all_dataset as CustomDatset
from utils_ import ridege_finetone_val
import models
import os,time
# Initialize the folder
os.makedirs("checkpoints",exist_ok=True)
os.makedirs("experiments",exist_ok=True)

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
if os.path.isfile("./checkpoints/1_hrnet.bth"):
    print(f"loadding the exit checkpoints ./checkpoints/1_hrnet.bth")
    model.load_state_dict(
    torch.load("./checkpoints/1_hrnet.bth"))
model.train()
# Creatr optimizer
optimizer = get_optimizer(args.configs, model)
lr_scheduler=lr_sche(config=args.configs["lr_strategy"])
last_epoch = args.configs['train']['begin_epoch']

# Load the datasets
train_dataset=CustomDatset(args.data_path,'train',split_name=args.split_name)
# val_dataset=CustomDatset(args.data_path,'val',split_name=args.split_name)
val_dataset=ridege_finetone_val(args.data_path,split_name=args.split_name,split='val')

# Create the data loaders
train_loader = DataLoader(train_dataset, 
                          batch_size=24,
                          shuffle=True, num_workers=args.configs['num_works'])
val_loader = DataLoader(val_dataset,
                        # batch_size=args.configs['train']['batch_size'],
                        batch_size=24,
                        shuffle=False, num_workers=args.configs['num_works'])
print("There is {} patch size".format(args.configs['train']['batch_size']))
print(f"Train: {len(train_loader)} {len(train_dataset)}, Val: {len(val_loader)} {len(val_dataset)}")
# Set up the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
print(f"using {device} for training")
# Set up the optimizer, loss function, and early stopping

early_stop_counter = 0
best_val_loss = float('inf')
total_epoches = args.configs['train']['end_epoch']
# max_auc=0
# Training and validation loop
acc,max_auc  = fineone_val_epoch(model, val_loader, criterion, device)
print(max_auc)
for epoch in range(last_epoch, total_epoches):
    start_time = time.time()  # Record the start time of the epoch
    train_loss = train_epoch(model, optimizer, train_loader, criterion, device,lr_scheduler,epoch)
    acc,auc  = fineone_val_epoch(model, val_loader, criterion, device)
    
    end_time = time.time()  # Record the end time of the epoch
    elapsed_time = end_time - start_time  # Calculate the elapsed time
    elapsed_hours = elapsed_time / 3600  # Convert elapsed time to hours
    print(f"Epoch {epoch + 1}/{total_epoches}, "
          f"Train Loss: {train_loss:.6f}, Val acc: {acc:.6f}, auc: {auc:.6f} "
          f"Lr: {optimizer.state_dict()['param_groups'][0]['lr']:.6f}, "
          f"Time: {elapsed_hours:.2f} hours")
    # Update the learning rate if using ReduceLROnPlateau or CosineAnnealingLR
    # Early stopping
    if auc > max_auc:
        max_auc=auc
        early_stop_counter = 0
        torch.save(model.state_dict(),
                   os.path.join(args.save_dir,f"{args.split_name}_{args.configs['save_name']}"))
        print("Model saved as {}".format(os.path.join(args.save_dir,f"{args.split_name}_{args.configs['save_name']}")))
    if epoch==total_epoches-1:
        torch.save(model.state_dict(),
                   os.path.join(args.save_dir,f"{args.split_name}_last_{args.configs['save_name']}"))
        print("Model saved as {}".format(os.path.join(args.save_dir,f"{args.split_name}_{args.configs['save_name']}")))