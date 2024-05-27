import json
import numpy as np

# Load the data
with open('./experiments/shen_record.json') as f:
    record_original = json.load(f)

# Prepare to save the processed data
save_path = './experiments/result_shen_record.json'
record = {}

# Process each model
for model_name, parameters in record_original.items():
    best_auc = -1
    best_result = None

    # Calculate mean and std for each configuration
    for param, results in parameters.items():
        metrics = {'Accuracy': [], 'AUC': [], 'Recall': []}
        
        # Collect all metric results across splits
        for split, split_results in results.items():
            metrics['Accuracy'].append(float(split_results['Accuracy']))
            metrics['AUC'].append(float(split_results['AUC']))
            metrics['Recall'].append(float(split_results['Recall']))

        # Calculate mean and std for each metric
        metric_means = {metric: np.mean(values) for metric, values in metrics.items()}
        metric_stds = {metric: np.std(values) for metric, values in metrics.items()}

        # Check if this is the best result based on AUC
        if metric_means['AUC'] > best_auc:
            best_auc = metric_means['AUC']
            best_result = {
                'AUC': [metric_means['AUC'], metric_stds['AUC']],
                'Accuracy': [metric_means['Accuracy'], metric_stds['Accuracy']],
                'recall_pos': [metric_means['Recall'], metric_stds['Recall']]
            }
        elif metric_means['AUC'] == best_auc:
            # If AUCs are the same, choose the one with smaller std
            if best_result['AUC'][1] > metric_stds['AUC']:
                best_result = {
                    'AUC': [metric_means['AUC'], metric_stds['AUC']],
                    'Accuracy': [metric_means['Accuracy'], metric_stds['Accuracy']],
                    'recall_pos': [metric_means['Recall'], metric_stds['Recall']]
                }

    # Format the results with four decimal places
    record[model_name] = {
        'AUC': [f"{best_result['AUC'][0]:.4f}", f"{best_result['AUC'][1]:.4f}"],
        'Accuracy': [f"{best_result['Accuracy'][0]:.4f}", f"{best_result['Accuracy'][1]:.4f}"],
        'recall_pos': [f"{best_result['recall_pos'][0]:.4f}", f"{best_result['recall_pos'][1]:.4f}"]
    }

# Save the processed data
with open(save_path, 'w') as f:
    json.dump(record, f, indent=4)
