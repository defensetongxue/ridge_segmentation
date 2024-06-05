import json
import os
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

# Define paths
hvd_path = './experiments/hvd.json'
nohvd_path = './experiments/nohvd.json'
path_dir = './experiments'
font_path = './arial.ttf'

# Ensure the output directory exists
output_dir = os.path.join(path_dir, 'HVD')
os.makedirs(output_dir, exist_ok=True)

# Load custom font
custom_font = FontProperties(fname=font_path)

# Function to load data from JSON file
def load_data(filepath):
    with open(filepath, 'r') as file:
        data = json.load(file)
    return data

# Load both datasets
hvd_data = load_data(hvd_path)
nohvd_data = load_data(nohvd_path)

# Metrics to plot
metrics = ['accuracy', 'auc', 'recall']

# Plotting
for metric in metrics:
    plt.figure(figsize=(10, 6))
    # Extract and plot data for each file
    epochs = sorted(hvd_data.keys(), key=int)  # Assuming epoch keys are sortable and all epochs are present in both
    hvd_values = [hvd_data[epoch][metric] for epoch in epochs]
    nohvd_values = [nohvd_data[epoch][metric] for epoch in epochs]

    plt.plot(epochs, hvd_values, label='HVD', color='red')
    plt.plot(epochs, nohvd_values, label='Ours', color='blue')
    plt.title(f'{metric.capitalize()} Over Epochs', fontproperties=custom_font, fontsize=40)
    plt.xlabel('Epoch', fontproperties=custom_font, fontsize=24)
    plt.ylabel(metric.capitalize(), fontproperties=custom_font, fontsize=24)
    plt.legend(loc='upper right')
    plt.grid(True)

    # Set custom x-axis ticks
    plt.xticks(range(0, max(map(int, epochs)) + 1, 10), fontproperties=custom_font, fontsize=24)
    plt.yticks(fontproperties=custom_font, fontsize=24)

    # Save the figure
    plt.savefig(os.path.join(output_dir, f'{metric}.png'), dpi=300, bbox_inches='tight')
    plt.close()

print("All plots have been generated and saved.")
