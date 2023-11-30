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

class Metrics:
    def __init__(self,header="Main" ):
        self.reset()
        self.header=header
    def reset(self):
        self.image_recall = 0
        self.image_acc = 0
        self.image_auc = 0
        self.stage_acc = 0
        self.stage_auc = 0
        self.confu_matrix = np.zeros((4, 4))
        self.recall_1 = 0
        self.recall_2 = 0
        self.recall_3 = 0
        self.positive_recall = 0
        
    def update(self, ridge_preds, ridge_labels, stage_preds, stage_labels,stage_probs):
        # Update image-level metrics
        self.image_recall = calculate_recall(ridge_labels, ridge_preds)
        self.image_acc = accuracy_score(ridge_labels, ridge_preds)
        self.image_auc = roc_auc_score(ridge_labels, ridge_preds)

        # Manually create confusion matrix for stage predictions
        num_classes = 4  # Adjust based on your number of classes
        self.confu_matrix = np.zeros((num_classes, num_classes))
        for true_label, pred_label in zip(stage_labels, stage_preds):
            self.confu_matrix[true_label, pred_label] += 1

        # Calculate recall for each class and positive recall
        self.recall_1 = self.confu_matrix[1, 1] / np.sum(self.confu_matrix[1, :])
        self.recall_2 = self.confu_matrix[2, 2] / np.sum(self.confu_matrix[2, :])
        self.recall_3 = self.confu_matrix[3, 3] / np.sum(self.confu_matrix[3, :])
        positive_correct = np.sum(self.confu_matrix[1:, 1:])
        positive_total = np.sum(self.confu_matrix[1:, :])
        self.positive_recall = positive_correct / positive_total

        # Calculate stage accuracy and AUC
        self.stage_acc = accuracy_score(stage_labels, stage_preds)
        self.stage_auc = roc_auc_score(stage_labels, stage_probs, multi_class='ovo')

    def __str__(self):
        return (f"[{self.header}] "
                f"Image - Acc: {self.image_acc:.4f}, AUC: {self.image_auc:.4f}, Recall: {self.image_recall:.4f}\n"
                f"[{self.header}] "
                f"Stage - Acc: {self.stage_acc:.4f}, AUC: {self.stage_auc:.4f}, "
                f"Recall1: {self.recall_1:.4f}, Recall2: {self.recall_2:.4f}, Recall3: {self.recall_3:.4f}, PosRecall: {self.positive_recall:.4f}\n"
                f"Confusion Matrix:\n{self.confu_matrix}")

    def _store(self, key, split_name, save_epoch, save_path='./record.json'):
        res = {
            "image_accuracy": round(self.image_acc, 4),
            "image_auc": round(self.image_auc, 4),
            "image_recall": round(self.image_recall, 4),
            "stage_accuracy": round(self.stage_acc, 4),
            "stage_auc": round(self.stage_auc, 4),
            "stage_recall_1": round(self.recall_1, 4),
            "stage_recall_2": round(self.recall_2, 4),
            "stage_recall_3": round(self.recall_3, 4),
            "positive_recall": round(self.positive_recall, 4),
            "confusion_matrix": self.confu_matrix.tolist(),
            "save_epoch": save_epoch
        }

        if os.path.exists(save_path):
            with open(save_path, 'r') as file:
                existing_data = json.load(file)
        else:
            existing_data = {}

        if key not in existing_data:
            existing_data[key] = {}
        existing_data[key][split_name] = res

        with open(save_path, 'w') as file:
            json.dump(existing_data, file, indent=4)