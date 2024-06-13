import json
import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties

# Load JSON data
# json_path = './experiments/result_shen_record.json'
json_path = './experiments/result_foshan_record.json'
with open(json_path, 'r') as file:
    data = json.load(file)

# Configuration
# t_bar = False
t_bar = True
font_path = './arial.ttf'
show_number = True
colors = [(51/255, 57/255, 91/255), (93/255, 116/255, 162/255), (196/255, 216/255, 242/255), (124/255, 40/255, 43/255)]
save_dir = './experiments/new_figure'
os.makedirs(save_dir, exist_ok=True)
font_size = 16

# Metrics to plot
metrics = ['AUC', 'Accuracy', 'recall_pos']
model_names = list(data.keys())

# Load font
font = FontProperties(fname=font_path, size=font_size)

# Plot figures
for metric in metrics:
    fig, ax = plt.subplots(figsize=(5, 6.5))
    means = []
    stds = []
    for model in model_names:
        mean, std = map(float, data[model][metric])
        means.append((mean - 0.6) / 0.4)
        stds.append(std / 0.4)  # Scale std to match adjusted mean
    
    bars = ax.bar(model_names, means, color=colors)
    ax.set_ylim(0, 1)
    ax.set_yticks(np.linspace(0, 1, 5))
    ax.set_yticklabels(np.round(0.6 + np.linspace(0, 1, 5) * 0.4, 2))
    
    # Show t-bar if required
    if t_bar:
        for bar, std in zip(bars, stds):
            bar_height = bar.get_height()
            # Draw the entire t-bar with both upper and lower parts
            ax.errorbar(bar.get_x() + bar.get_width() / 2, bar_height, yerr=std, fmt='none', ecolor='black', elinewidth=1, capsize=5, capthick=1)
    
    # Show mean value if required
    if show_number:
        for bar, mean in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, 0.02, f'{0.6 + mean * 0.4:.3f}', ha='center', va='bottom', fontsize=font_size, fontproperties=font, color='white')
    
    # Rotate x labels and set font
    plt.xticks(rotation=45, fontproperties=font)
    plt.yticks(fontproperties=font)
    
    # Set title, adjust for Recall Positive
    title = metric.replace('_', ' ').title()
    if metric == 'recall_pos':
        title = 'Recall Positive'
    ax.set_title(title, fontproperties=font)
    
    # Save figure
    fig.savefig(os.path.join(save_dir, f'{metric}.png'), bbox_inches='tight',dpi=300)

plt.close('all')
