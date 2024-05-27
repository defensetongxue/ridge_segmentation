import json
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.patches import Patch
from matplotlib.font_manager import FontProperties

def compress_scores(value, lower_bound=0.6):
    if value > lower_bound:
        return (value - lower_bound)
    return value

# Load metrics from JSON file
json_path = './experiments/result_foshan_record.json'
with open(json_path, 'r') as f:
    results = json.load(f)

# Metrics to plot
metrics = ['Accuracy', 'AUC', 'Recall']
model_names = list(results.keys())
colors = [(51/255, 57/255, 91/255), (93/255, 116/255, 162/255), (196/255, 216/255, 242/255), (124/255, 40/255, 43/255)]

# Initialize dictionaries to store best mean metrics across parameters for each model
best_mean_metrics = {model: {metric: 0 for metric in metrics} for model in model_names}
best_std_metrics = {model: {metric: 0 for metric in metrics} for model in model_names}
best_auc = {model: 0 for model in model_names}  # Best AUC for each model to compare parameters

for model in model_names:
    for parameter in results[model].keys():
        metric_values = {metric: [] for metric in metrics}
        
        for split in results[model][parameter]:
            for metric in metrics:
                metric_values[metric].append(float(results[model][parameter][split][metric]))
        
        # Calculate mean and std for current parameter
        mean_metrics = {metric: np.mean(values) for metric, values in metric_values.items()}
        std_metrics = {metric: np.std(values) for metric, values in metric_values.items()}
        
        # If current parameter has the highest AUC, store its metrics
        if mean_metrics['AUC'] > best_auc[model]:
            best_auc[model] = mean_metrics['AUC']
            best_mean_metrics[model] = mean_metrics
            best_std_metrics[model] = std_metrics

bar_width = 0.6  # Bar width

experiments_dir = './experiments'
os.makedirs(experiments_dir, exist_ok=True)

for metric in metrics:
    fig, ax = plt.subplots(figsize=(2, 2.5))
    positions = np.arange(len(model_names))

    for i, model in enumerate(model_names):
        mean_value = best_mean_metrics[model][metric]
        std_value = best_std_metrics[model][metric]
        compressed_mean = compress_scores(mean_value)  # Apply compression
        
        ax.bar(positions[i], compressed_mean, bar_width, label=model, color=colors[i], yerr=std_value, capsize=5, align='center')

    # Customize y-axis
    y_ticks = np.linspace(0, max([compress_scores(1)]), 5)
    y_tick_labels = [f"{0.6 + tick * (0.4 / max(y_ticks)):.2f}" for tick in y_ticks]
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_tick_labels)

    ax.set_xticks(positions)
    ax.set_xticklabels(model_names, rotation=45, ha="right")

    ax.set_title(f'{metric}')

    plt.savefig(os.path.join(experiments_dir, f'{metric}.jpg'), dpi=300, bbox_inches='tight')
    plt.close(fig)

# Create a legend
fig_legend = plt.figure(figsize=(3, 2))
ax_legend = fig_legend.add_subplot(111)
legend_elements = [Patch(facecolor=colors[i], label=model_names[i]) for i in range(len(model_names))]
legend = ax_legend.legend(handles=legend_elements, loc='center')
ax_legend.axis('off')
fig_legend.savefig(os.path.join(experiments_dir, 'legend.jpg'), dpi=300, bbox_inches='tight')
plt.close(fig_legend)
