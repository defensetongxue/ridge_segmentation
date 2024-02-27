import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
from matplotlib.patches import Patch

# Load the JSON file
with open('results.json', 'r') as f:
    results = json.load(f)

# Define the model names and splits if they are dynamic
model_names = list(results.keys())
splits = list(results[model_names[0]].keys())
metrics = ['Accuracy', 'AUC', 'Recall']
# Organize data for easy access
metric_values = {metric: {model: [] for model in model_names} for metric in metrics}

for model in model_names:
    for split in splits:
        for metric in metrics:
            metric_values[metric][model].append(float(results[model][split][metric]))
colors = [(51/255, 57/255, 91/255), (93/255, 116/255, 162/255), (196/255, 216/255, 242/255)]
font_path = './arial.ttf'

for metric in metrics:
    fig, ax = plt.subplots()
    width = 0.2  # Width of the bars
    model_positions = np.arange(len(splits))  # Position of groups

    for i, model in enumerate(model_names):
        values = metric_values[metric][model]
        ax.bar(model_positions + i*width, values, width, label=model, color=colors[i])

    # Adding T-test annotations if needed
    # Example: comparing model 1 and model 2 for the first split
    # t_stat, p_val = ttest_ind(metric_values[metric][model_names[0]], metric_values[metric][model_names[1]])
    # ax.text(x, y, f"T={t_stat:.2f}, p={p_val:.3f}", fontsize=9)  # Position (x, y) needs to be adjusted

    ax.set_ylabel(metric)
    ax.set_title(f'{metric} by model and split')
    ax.set_xticks(model_positions + width)
    ax.set_xticklabels(splits)
    ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1))

    # Customize font
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontname('Arial')
        label.set_fontsize(12)

    fig.tight_layout()
    plt.show()
