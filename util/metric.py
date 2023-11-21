import numpy as np
import torch
from sklearn.metrics import accuracy_score, roc_auc_score
from collections import Counter
import json,os
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

def calculate_dice_iou(predict, target):
    smooth = 1e-5
    intersection = (predict * target).sum()
    union = predict.sum() + target.sum()
    dice = (2. * intersection + smooth) / (union + smooth)
    iou = (intersection + smooth) / (union - intersection + smooth)
    return dice.item(), iou.item()

class Metrics:
    def __init__(self,header="Main" ):
        self.reset()
        self.header=header
    def reset(self):
        # Image-level metrics
        self.image_recall = 0
        self.image_acc = 0
        self.image_auc = 0

        # Pixel-level metrics
        self.pixel_acc = 0
        self.pixel_auc = 0
        self.dice = 0
        self.iou = 0

    def update(self,pixel_preds,pixel_labels, image_preds,image_labels):
        self.image_recall = calculate_recall(image_labels, image_preds)
        self.image_acc = accuracy_score(image_labels, image_preds)
        self.image_auc = roc_auc_score(image_labels, image_preds)
        self.pixel_acc = accuracy_score(pixel_labels, pixel_preds > 0.5)
        self.pixel_auc = roc_auc_score(pixel_labels, pixel_preds)
        self.dice, self.iou = calculate_dice_iou(torch.tensor(pixel_preds > 0.5, dtype=torch.float32), torch.tensor(pixel_labels, dtype=torch.float32))

    def __str__(self):
        return (f"[{self.header}] "
                f"Image - Acc: {self.image_acc:.4f}, AUC: {self.image_auc:.4f}, Recall: {self.image_recall:.4f}\n"
                f"Pixel - Acc: {self.pixel_acc:.4f}, AUC: {self.pixel_auc:.4f}, Dice: {self.dice:.4f}, IOU: {self.iou:.4f}")

    def _store(self, key, split_name, save_epoch, param, save_path='./record.json'):
        res = {
            "image_accuracy": round(self.image_acc, 4),
            "image_auc": round(self.image_auc, 4),
            "image_recall": round(self.image_recall, 4),
            "pixel_accuracy": round(self.pixel_acc, 4),
            "pixel_auc": round(self.pixel_auc, 4),
            "dice": round(self.dice, 4),
            "iou": round(self.iou, 4),
            "save_epoch": save_epoch
        }


        # Check if the file exists and load its content if it does
        if os.path.exists(save_path):
            with open(save_path, 'r') as file:
                existing_data = json.load(file)
        else:
            existing_data = {}

        # Append the new data
        if key not in existing_data:
            existing_data[key]={
                "param":param
            }
        existing_data[key][split_name]=res

        # Save the updated data back to the file
        with open(save_path, 'w') as file:
            json.dump(existing_data, file, indent=4)
            
            