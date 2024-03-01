import json
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.patches import Patch
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import os

def compress_scores(value, lower_bound=0.6):
    """Compresses score values above a threshold for enhanced visual contrast."""
    if value > lower_bound:
        # Apply a non-linear transformation or adjust as needed
        return  (value - lower_bound)
    return value

# Assuming you have the path to your desired font
font_path = 'arial.ttf'  # Update this path
font_prop = FontProperties(fname=font_path, size=6)

# Load metrics from JSON file
json_path = 'experiments/record.json'
with open(json_path, 'r') as f:
    results = json.load(f)

# Metrics to plot
metrics = ['Accuracy', 'AUC', 'Recall']
model_names = list(results.keys())
colors = [(51/255, 57/255, 91/255),(93/255, 116/255, 162/255), (196/255, 216/255, 242/255),(124/255,40/255,43/255)]

# Initialize dictionaries to store means and standard deviations
mean_metrics = {metric: {model: [] for model in model_names} for metric in metrics}
std_metrics = {metric: {model: [] for model in model_names} for metric in metrics}

for metric in metrics:
    for model in model_names:
        metric_values = [float(results[model][split][metric]) for split in results[model]]
        # Store mean and standard deviation for each metric and model
        mean_metrics[metric][model] = np.mean(metric_values)
        std_metrics[metric][model] = np.std(metric_values)

bar_width = 0.6  # Adjust bar width here

experiments_dir = './experiments'
os.makedirs(experiments_dir, exist_ok=True)

for metric in metrics:
    fig, ax = plt.subplots(figsize=(2, 2.5))
    positions = np.arange(len(model_names))

    for i, model in enumerate(model_names):
        mean_value = mean_metrics[metric][model]  # Assume these are already calculated
        std_value = std_metrics[metric][model]    # Assume these are already calculated
        compressed_mean = compress_scores(mean_value)  # Your compress_scores function
        
        ax.bar(positions[i], compressed_mean, bar_width, label=model, color=colors[i], yerr=std_value, capsize=5, align='center')

    # Customization for y-axis to reflect the real score range
    y_ticks = np.linspace(0, max([compress_scores(1)]), 5)
    y_tick_labels = [f"{0.6 + tick * (0.4 / max(y_ticks)):.2f}" for tick in y_ticks]
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_tick_labels, fontproperties=font_prop)

    ax.set_xticks(positions)
    ax.set_xticklabels(model_names, fontproperties=font_prop)

    # Set the title with custom font properties
    ax.set_title(f'{metric}', fontproperties=font_prop)

    # If you have axis labels, set them with custom font properties as well
    # ax.set_xlabel('Your X-Axis Label', fontproperties=font_prop)
    # ax.set_ylabel('Your Y-Axis Label', fontproperties=font_prop)

    plt.savefig(os.path.join(experiments_dir, f'{metric}.jpg'), dpi=300, bbox_inches='tight')
    plt.close(fig)

# Create a figure for the legend
fig_legend = plt.figure(figsize=(3, 2))
ax_legend = fig_legend.add_subplot(111)
legend_elements = [Patch(facecolor=colors[i], label=model_names[i]) for i in range(len(model_names))]
legend = ax_legend.legend(handles=legend_elements, loc='center')
ax_legend.axis('off')

# Save just the legend
fig_legend.savefig(os.path.join(experiments_dir, 'legend.jpg'), dpi=300, bbox_inches='tight')
plt.close(fig_legend)
