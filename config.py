import argparse
import json

def get_config():
    parser = argparse.ArgumentParser()
    
    # cleansing
    parser.add_argument('--data_path', type=str, default='../autodl-tmp/dataset_ROP',
                        help='Path to the directory where the processed datasets are stored.')
    parser.add_argument('--generate_ridge_diffusion', action='store_true',
                        help='Flag to indicate if ridge coordinates should be generated from the source JSON file.')
    parser.add_argument('--generate_mask', action='store_true',
                        help='Flag to indicate if masks should be generated. this will crop the orignal ridge diffusion to patchtigy image.')
    
    # split
    parser.add_argument('--split_name', type=str, default='clr_1',
                        help='Specify the dataset split to use for training and evaluation.')
    
    # Model
    parser.add_argument('--patch_size', type=int, default=400,
                        help='Size of the patches extracted from the images.')
    parser.add_argument('--stride', type=int, default=200,
                        help='Stride used for extracting patches from the images.')
    parser.add_argument('--ridge_seg_number', type=int, default=8,
                        help='Number of segments for ridge detection.')
    
    # train and test
    parser.add_argument('--save_dir', type=str, default="./checkpoints",
                        help='Directory to save the best model during training.')
    parser.add_argument('--result_path', type=str, default="experiments",
                        help='Path to save the visualization results or the trained PyTorch model.')
    parser.add_argument('--from_checkpoint', type=str, default="",
                        help='Path to an existing checkpoint to resume training from.')
    
    parser.add_argument('--lr', type=float, default=5e-4,
                        help='Learning rate for the optimizer.')
    parser.add_argument('--wd', type=float, default=5e-4,
                        help='Weight decay for the optimizer.')
    
    # config file 
    parser.add_argument('--cfg', type=str, default="./config_file/hrnet_w48.json",
                        help='Path to the experiment configuration file in JSON format.')
    
    args = parser.parse_args()
    
    # Merge args and config file 
    with open(args.cfg, 'r') as f:
        args.configs = json.load(f)
    
    return args
