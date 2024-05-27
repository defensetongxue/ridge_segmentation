import json
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import font_manager

# Adding the custom font to Matplotlib's font manager
font_path = './arial.ttf'
font_prop = font_manager.FontProperties(fname=font_path, size=14)
sns.set_theme(style="whitegrid")  # Enable grid first
sns.set_theme(style="ticks", rc={"axes.facecolor": "white", "grid.color": "white"})  # Now disable grid and set white background



def load_data(data_path, split_name):
    """ Load data from JSON files and segregate based on the provided split. """
    with open(os.path.join(data_path, 'annotations.json')) as f:
        data_dict = json.load(f)

    with open(os.path.join(data_path, 'split', split_name)) as f:
        used_dict = json.load(f)
        used_list = used_dict['test']+used_dict['train']+used_dict['val']

    list_0, list_1 = [], []
    missing_ridge_seg = []
    
    for image_name in used_list:
        data = data_dict[image_name]
        if 'ridge_seg' not in data:
            missing_ridge_seg.append(image_name)
            continue
        val = data['ridge_seg']['max_val']
        if data['stage'] == 0:
            list_0.append(val)
        else:
            list_1.append(val)
    
    return list_0, list_1, missing_ridge_seg

def plot_values(list_0, list_1):
    """ Plot the distribution of values for the two lists using the specified font. """
    color1 = (93/255, 116/255, 162/255)  # RGB color converted to Matplotlib format
    color2 = (69/255, 51/255, 112/255)  # RGB color converted to Matplotlib format

    # Plotting with the specified font
    plt.figure()
    ax = sns.kdeplot(list_0, fill=True, color=color1)
    sns.kdeplot(list_1, fill=True, color=color2)
    ax.set_xlabel("Value", fontproperties=font_prop)
    ax.set_ylabel("Density", fontproperties=font_prop)
    ax.set_title("Distribution of Max Value in Positive and Negative Images", fontproperties=font_prop)
    plt.savefig('./experiments/foshan_dense.png', dpi=300)
    plt.show()

data_path = '../autodl-tmp/dataset_ROP'
split_name = 'clr_1.json'
list_0, list_1, missing = load_data(data_path, split_name)
plot_values(list_0, list_1)
if missing:
    print(f"The following images do not have 'ridge_seg' data: {missing}")
