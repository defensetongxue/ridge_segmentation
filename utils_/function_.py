import torch,math
from torch import optim
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from sklearn.metrics import accuracy_score, roc_auc_score
def to_device(x, device):
    if isinstance(x, tuple):
        return tuple(to_device(xi, device) for xi in x)
    elif isinstance(x,list):
        return [to_device(xi,device) for xi in x]
    else:
        return x.to(device)
def calculate_recall(labels, preds):
    """
    Calculate recall for class 1 in a binary classification task.
    
    Args:
    labels (np.array): Array of true labels.
    preds (np.array): Array of predicted labels.
    
    Returns:
    float: Recall for class 1.
    """
    # Ensure labels and predictions are numpy arrays
    labels = np.array(labels)
    preds = np.array(preds)

    # Calculate True Positives and False Negatives
    true_positives = np.sum((labels == 1) & (preds == 1))
    false_negatives = np.sum((labels == 1) & (preds == 0))

    # Calculate recall
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    return recall
def train_epoch(model, optimizer, train_loader, loss_function, device,lr_scheduler,epoch):
    model.train()
    running_loss = 0.0
    batch_length=len(train_loader)
    for data_iter_step,(inputs, targets, meta) in enumerate(train_loader):
        # Moving inputs and targets to the correct device
        lr_scheduler.adjust_learning_rate(optimizer,epoch+(data_iter_step/batch_length))
        inputs = to_device(inputs, device)
        targets = to_device(targets, device)

        optimizer.zero_grad()

        # Assuming your model returns a tuple of outputs
        outputs = model(inputs)

        # Assuming your loss function can handle tuples of outputs and targets
        loss = loss_function(outputs, targets)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    
    return running_loss / len(train_loader)

def train_total_epoch(model, optimizer, train_loader, loss_function, device,lr_scheduler,epoch):
    model.train()
    running_loss = 0.0
    batch_length=len(train_loader)
    for data_iter_step,(inputs, targets, meta) in enumerate(train_loader):
        # Moving inputs and targets to the correct device
        lr_scheduler.adjust_learning_rate(optimizer,epoch+(data_iter_step/batch_length))
        inputs = to_device(inputs, device)
        targets = to_device(targets, device)

        optimizer.zero_grad()

        # Assuming your model returns a tuple of outputs
        outputs = model(inputs)
        outputs = torch.sigmoid(outputs)
        outputs=outputs.flatten(1,3)
        # print(outputs.shape)
        max_vals,_ =torch.max(outputs,dim=1) # bc,1
        # print(max_vals.shape)
         # Construct a bc, 2 output, each row is [max_val, 1-max_val]
        output_class = torch.stack([ 1 - max_vals,max_vals], dim=1)
        # print(output_class.shape)
        # Compute loss; we assume targets are in bc shape
        # CrossEntropyLoss expects inputs of shape (N, C) and targets of shape (N)
        loss = loss_function(output_class, targets.long())

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    
    return running_loss / len(train_loader)
def val_epoch(model, val_loader, loss_function, device):
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for inputs, targets,meta in val_loader:
            inputs = to_device(inputs, device)
            targets = to_device(targets, device)

            outputs = model(inputs)
            loss = loss_function(outputs, targets)

            running_loss += loss.item()

    return running_loss / len(val_loader)

def fineone_val_epoch(model, val_loader, loss_function, device):
    model.eval()
    running_loss = 0.0
    predict=[]
    labels=[]
    with torch.no_grad():
        for inputs, targets,meta in val_loader:
            inputs = to_device(inputs, device)

            outputs = model(inputs).cpu()
            outputs=torch.sigmoid(outputs)
            ridge_mask=torch.where(outputs.squeeze(1)>0.5,1,0).flatten(1,2)
            ridge_mask=torch.sum(ridge_mask,dim=1)
            predict_label=torch.where(ridge_mask>5,1,0).tolist()
            predict.extend(predict_label)
            labels.extend(targets)
    labels=np.array(labels)
    predict=np.array(predict)
    recall=calculate_recall(labels,predict)
    acc = accuracy_score(labels, predict)
    auc = roc_auc_score(labels, predict)
    return acc,auc,recall

def get_instance(module, class_name, *args, **kwargs):
    cls = getattr(module, class_name)
    instance = cls(*args, **kwargs)
    return instance

def get_optimizer(cfg, model):
    optimizer = None
    if cfg['train']['optimizer'] == 'sgd':
        optimizer = optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=cfg['train']['lr'],
            momentum=cfg['train']['momentum'],
            weight_decay=cfg['train']['wd'],
            nesterov=cfg['train']['nesterov']
        )
    elif cfg['train']['optimizer'] == 'adam':
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=cfg['train']['lr']
        )
    elif cfg['train']['optimizer'] == 'rmsprop':
        optimizer = optim.RMSprop(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=cfg['train']['lr'],
            momentum=cfg['train']['momentum'],
            weight_decay=cfg['train']['wd'],
            alpha=cfg['train']['rmsprop_alpha'],
            centered=cfg['train']['rmsprop_centered']
        )
    else:
        raise
    return optimizer
def get_lr_scheduler(optimizer, cfg):
    if cfg['method'] == 'reduce_plateau':
        lr_scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            patience=cfg['reduce_plateau_patience'],
            factor=cfg['reduce_plateau_factor'],
            cooldown=cfg['cooldown'],
            verbose=False
        )
    elif cfg['method'] == 'cosine_annealing':
        lr_scheduler = CosineAnnealingLR(optimizer, T_max=cfg['cosine_annealing_T_max'])
    elif cfg['method'] == 'constant':
        lr_scheduler = None  # No learning rate scheduling for constant LR
    else:
        raise ValueError("Invalid learning rate scheduling method")
    
    return lr_scheduler

class lr_sche():
    def __init__(self,config):
        self.warmup_epochs=config["warmup_epochs"]
        self.lr=config["lr"]
        self.min_lr=config["min_lr"]
        self.epochs=config['epochs']
    def adjust_learning_rate(self,optimizer, epoch):
        """Decay the learning rate with half-cycle cosine after warmup"""
        if epoch < self.warmup_epochs:
            lr = self.lr * epoch / self.warmup_epochs
        else:
            lr = self.min_lr + (self.lr  - self.min_lr) * 0.5 * \
                (1. + math.cos(math.pi * (epoch - self.warmup_epochs) / (self.epochs - self.warmup_epochs)))
        for param_group in optimizer.param_groups:
            if "lr_scale" in param_group:
                param_group["lr"] = lr * param_group["lr_scale"]
            else:
                param_group["lr"] = lr
        return lr