import torch,math
from torch import optim
import numpy as np
from .metric import Metrics
def to_device(x, device):
    if isinstance(x, tuple):
        return tuple(to_device(xi, device) for xi in x)
    elif isinstance(x,list):
        return [to_device(xi,device) for xi in x]
    else:
        return x.to(device)
    


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

def val_epoch(model, val_loader, loss_function, device,metirc:Metrics):
    model.eval()
    running_loss = 0.0
    pixel_preds = []
    pixel_labels = []
    image_preds = []
    image_labels = []
    with torch.no_grad():
        for inputs, targets, meta in val_loader:
            inputs = inputs.to(device)

            outputs = model(inputs)
            loss = loss_function(outputs, targets)
            running_loss += loss.item()
            
            outputs=outputs.cpu()
            outputs = torch.sigmoid(outputs)
            outputs_flat = outputs.view(-1)
            targets_flat = targets.view(-1)

            # For pixel-level metrics
            pixel_preds.extend(outputs_flat.numpy())
            pixel_labels.extend(targets_flat.numpy())

            # For image-level classification
            ridge_mask = torch.where(outputs.squeeze(1) > 0.5, 1, 0).flatten(1, 2)
            ridge_mask_sum = torch.sum(ridge_mask, dim=1)
            predict_label = torch.where(ridge_mask_sum > 5, 1, 0).tolist()
            image_preds.extend(predict_label)
            image_labels.extend(targets)

    # Convert to NumPy arrays for metric calculation
    pixel_preds = np.array(pixel_preds)
    pixel_labels = np.array(pixel_labels)
    image_preds = np.array(image_preds)
    image_labels = np.array(image_labels)

    metirc.update(pixel_preds,pixel_labels,image_preds,image_labels)
    return running_loss / len(val_loader),  metirc

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
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=cfg['train']['lr'],weight_decay=cfg['train']['wd']
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