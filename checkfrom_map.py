import os
import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from PIL import Image
from sklearn.metrics import accuracy_score, roc_auc_score
from utils_.function_ import calculate_recall

class CropPadding:
    def __init__(self, box=(80, 0, 1570, 1200)):
        self.box = box

    def __call__(self, img):
        return img.crop(self.box)

def fusion_the_map(data_dict, split_list, masks, judges):
    # Initialize matrices to store predictions for each combination of thresholds
    pred_matrix = np.zeros((len(masks), len(judges), len(split_list)), dtype=int)

    # Process each image only once
    for idx, image_name in enumerate(split_list):
        data = data_dict[image_name]
        ridge_seg = Image.open(data['ridge_seg']["ridge_seg_path"]).convert('L')
        ridge_seg = CropPadding()(ridge_seg)
        tar_size = ridge_seg.size
        ridge_seg = np.array(ridge_seg, dtype=np.float32) / 255
        retfound_embed = Image.open(data['retfound_embedding']).convert('L').resize(tar_size, resample=Image.NEAREST)
        retfound_embed = np.array(retfound_embed) / 255

        # Compute predictions for all threshold combinations
        for i, mask in enumerate(masks):
            mask_applied = np.where(retfound_embed > mask, 1, 0)
            ridge_seg_masked = ridge_seg * mask_applied

            for j, judge in enumerate(judges):
                pred_matrix[i, j, idx] = 1 if np.max(ridge_seg_masked) > judge else 0

    # Calculate metrics for each combination of thresholds
    labels = np.array([1 if 'ridge' in data_dict[image_name] else 0 for image_name in split_list])
    acc_values = np.zeros((len(masks), len(judges)))
    auc_values = np.zeros((len(masks), len(judges)))
    recall_values = np.zeros((len(masks), len(judges)))

    for i in range(len(masks)):
        for j in range(len(judges)):
            preds = pred_matrix[i, j, :]
            acc_values[i, j] = accuracy_score(labels, preds)
            auc_values[i, j] = roc_auc_score(labels, preds)
            recall_values[i, j] = calculate_recall(labels, preds)

    return acc_values, auc_values, recall_values

def plot_metric(metric_values, masks, judges, title, save_path):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    X, Y = np.meshgrid(judges, masks)  # Swap masks and judges here
    surf = ax.plot_surface(X, Y, metric_values, cmap='viridis', edgecolor='none')

    ax.set_xlabel('Mask Threshold')
    ax.set_ylabel('Judge Threshold')
    ax.set_zlabel(title)

    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    plt.title(title)
    plt.savefig(save_path)
    plt.close()

def find_best_thresholds(metric_values, masks, judges, metric_name):
    max_value = np.max(metric_values)
    max_index = np.unravel_index(np.argmax(metric_values, axis=None), metric_values.shape)
    best_mask = masks[max_index[0]]
    best_judge = judges[max_index[1]]
    print(f"Best {metric_name}: {max_value:.2f} at Mask Threshold: {best_mask}, Judge Threshold: {best_judge}")

if __name__ == "__main__":
    from config import get_config
    args = get_config()
    with open(os.path.join(args.data_path, 'split', f'{args.split_name}.json'), 'r') as f:
        split_list = json.load(f)['test']
    with open(os.path.join(args.data_path, 'annotations.json'), 'r') as f:
        data_dict = json.load(f)

    masks = np.arange(0.0, 1.00001, 0.01)
    judges = np.arange(0.4, 0.6, 0.01)

    acc_values, auc_values, recall_values = fusion_the_map(data_dict, split_list, masks, judges)

    find_best_thresholds(acc_values, masks, judges, "Accuracy")
    find_best_thresholds(auc_values, masks, judges, "AUC")
    find_best_thresholds(recall_values, masks, judges, "Recall")

    # Continue to plot without annotations
    plot_metric(acc_values, masks, judges, 'Accuracy', 'experiments/acc.jpg')
    plot_metric(auc_values, masks, judges, 'AUC', 'experiments/auc.jpg')
    plot_metric(recall_values, masks, judges, 'Recall', 'experiments/recall.jpg')