import json
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.patches import Patch
from matplotlib.font_manager import FontProperties

def compress_scores(value, lower_bound=0.6):
    """Compresses score values above a threshold for enhanced visual contrast."""
    if value > lower_bound:
        return (value - lower_bound) * 2  # Example compression
    return 0  # Adjust as needed

# Update this path to the font you wish to use
font_path = 'arial.ttf'
font_prop = FontProperties(fname=font_path, size=6)

# Load metrics from JSON file for a single split
json_path = 'experiments/shen_record.json'
with open(json_path, 'r') as f:
    results = json.load(f)

# Define metrics, model names, and colors
metrics = ['Accuracy', 'AUC', 'Recall']
model_names = list(results.keys())
colors = [(51/255, 57/255, 91/255), (93/255, 116/255, 162/255), (196/255, 216/255, 242/255), (124/255, 40/255, 43/255)]

experiments_dir = './experiments'
os.makedirs(experiments_dir, exist_ok=True)

# Iterate over each metric to plot
for metric in metrics:
    fig, ax = plt.subplots(figsize=(2, 2.5))  # Adjust figure size as needed
    positions = np.arange(len(model_names))
    bar_width = 0.6  # Adjust bar width here

    for i, model in enumerate(model_names):
        # Directly use the metric values from the results
        metric_value = float(results[model][metric])
        compressed_metric_value = compress_scores(metric_value)
        
        # Plot compressed metric value
        ax.bar(positions[i], compressed_metric_value, bar_width, label=model, color=colors[i], align='center')

    # Adjust y-axis to reflect the real score range from 0.6 to 1 after compression
    y_ticks = np.linspace(0, compress_scores(1), 5)
    y_tick_labels = [f"{0.6 + tick/2:.2f}" for tick in y_ticks]  # Decompressing back to original scale
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_tick_labels, fontproperties=font_prop)

    ax.set_xticks(positions)
    ax.set_xticklabels(model_names, fontproperties=font_prop,  ha="center")
    # ax.set_xticklabels(model_names, fontproperties=font_prop, rotation=45, ha="right")
    ax.set_title(f'{metric}', fontproperties=font_prop)

    plt.tight_layout()
    plt.savefig(os.path.join(experiments_dir, f'{metric}_single_split.jpg'), dpi=300, bbox_inches='tight')
    plt.close(fig)

# Creating a figure for the legend
fig_legend = plt.figure(figsize=(3, 2))
ax_legend = fig_legend.add_subplot(111)
legend_elements = [Patch(facecolor=colors[i], label=model_names[i]) for i in range(len(model_names))]
legend = ax_legend.legend(handles=legend_elements, loc='center', prop=font_prop)
ax_legend.axis('off')
fig_legend.savefig(os.path.join(experiments_dir, 'legend_shen.jpg'), dpi=300, bbox_inches='tight')
plt.close(fig_legend)
