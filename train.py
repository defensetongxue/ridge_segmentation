import torch
from torch.utils.data import DataLoader
from config import get_config
from utils_ import get_instance, train_epoch, val_epoch,get_optimizer,losses
from utils_ import ridge_segmentataion_dataset as CustomDatset
import models
import os
# Initialize the folder
os.makedirs("checkpoints",exist_ok=True)
os.makedirs("experiments",exist_ok=True)

# Parse arguments
args = get_config()

# Init the result file to store the pytorch model and other mid-result
result_path = args.result_path
os.makedirs(result_path,exist_ok=True)
print(f"the mid-result and the pytorch model will be stored in {result_path}")

# Create the model and criterion
model = get_instance(models, args.model)
criterion=get_instance(losses,args.configs.Loss)
if os.path.isfile(args.from_checkpoint):
    print(f"loadding the exit checkpoints {args.from_checkpoint}")
    model.load_state_dict(
    torch.load(args.from_checkpoint))
model.train()
# Creatr optimizer
optimizer = get_optimizer(args.configs, model)
last_epoch = args.configs.TRAIN.BEGIN_EPOCH
if isinstance(args.configs.TRAIN.LR_STEP, list):
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, args.configs.TRAIN.LR_STEP,
        args.configs.TRAIN.LR_FACTOR, last_epoch-1
    )
else:
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, args.configs.TRAIN.LR_STEP,
        args.configs.TRAIN.LR_FACTOR, last_epoch-1
    )

# Load the datasets
train_dataset=CustomDatset(args.path_tar,'train')
val_dataset=CustomDatset(args.path_tar,'val')
# Create the data loaders
train_loader = DataLoader(train_dataset, 
                          batch_size=args.configs.TRAIN.BATCH_SIZE_PER_GPU,
                          shuffle=True, num_workers=args.configs.WORKERS)
val_loader = DataLoader(val_dataset,
                        batch_size=args.configs.TRAIN.BATCH_SIZE_PER_GPU,
                        shuffle=False, num_workers=args.configs.WORKERS)
print(f"There is {args.configs.TRAIN.BATCH_SIZE_PER_GPU} patch size")
print(f"Train: {len(train_loader)}, Val: {len(val_loader)}")
# Set up the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
print(f"using {device} for training")
# Set up the optimizer, loss function, and early stopping

early_stop_counter = 0
best_val_loss = float('inf')
total_epoches=args.configs.TRAIN.END_EPOCH
# Training and validation loop
for epoch in range(last_epoch,total_epoches):

    train_loss = train_epoch(model, optimizer, train_loader, criterion, device)
    lr_scheduler.step()
    
    val_loss = val_epoch(model, val_loader, criterion, device)
    print(f"Epoch {epoch + 1}/{total_epoches}," 
          f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}," 
            f" Lr: {optimizer.state_dict()['param_groups'][0]['lr']:.6f}" )
    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        early_stop_counter = 0
        torch.save(model.state_dict(),(os.path.join(args.save_name)))
        print(f"Model saved as {args.save_name}")
    else:
        early_stop_counter += 1
        if early_stop_counter >= args.configs.TRAIN.EARLY_STOP:
            print("Early stopping triggered")
            break
