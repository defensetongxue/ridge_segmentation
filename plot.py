import json
import os
import matplotlib.pyplot as plt

# Define paths
hvd_path = './experiments/hvd.json'
nohvd_path = './experiments/nohvd.json'
path_dir = './experiments'

# Ensure the output directory exists
output_dir = os.path.join(path_dir, 'HVD')
os.makedirs(output_dir, exist_ok=True)

# Function to load data from JSON file
def load_data(filepath):
    with open(filepath, 'r') as file:
        data = json.load(file)
    return data

# Load both datasets
hvd_data = load_data(hvd_path)
nohvd_data = load_data(nohvd_path)

# Metrics to plot
metrics = ['image_accuracy', 'image_auc', 'image_recall']

# Plotting
for metric in metrics:
    plt.figure(figsize=(10, 6))
    # Extract and plot data for each file
    epochs = sorted(hvd_data.keys(), key=int)  # Assuming epoch keys are sortable and all epochs are present in both
    hvd_values = [hvd_data[epoch][metric] for epoch in epochs]
    nohvd_values = [nohvd_data[epoch][metric] for epoch in epochs]

    plt.plot(epochs, hvd_values, label='HVD', color='red')
    plt.plot(epochs, nohvd_values, label='Ours', color='blue')
    plt.title(f'{metric.capitalize()} Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel(metric.capitalize())
    plt.legend(loc='upper right')
    plt.grid(True)

    # Save the figure
    plt.savefig(os.path.join(output_dir, f'{metric}.png'), dpi=300, bbox_inches='tight')
    plt.close()

print("All plots have been generated and saved.")
