import argparse
from yacs.config import CfgNode as CN


_C = CN()
_C.GPUS = (0, )
_C.WORKERS = 16
_C.Loss='BCELoss'
# Cudnn related params
_C.CUDNN = CN()
_C.CUDNN.BENCHMARK = True
_C.CUDNN.DETERMINISTIC = False
_C.CUDNN.ENABLED = True

# common params for NETWORK
_C.MODEL = CN()
_C.MODEL.NAME = 'FR_UNet'
_C.MODEL.INIT_WEIGHTS = True
_C.MODEL.PRETRAINED = ''

# train
_C.TRAIN = CN()

_C.TRAIN.LR_FACTOR = 0.1
_C.TRAIN.LR_STEP = [30, 50]
_C.TRAIN.LR = 0.0001

_C.TRAIN.OPTIMIZER = 'adam'
_C.TRAIN.MOMENTUM = 0.0
_C.TRAIN.WD = 0.0
_C.TRAIN.NESTEROV = False

_C.TRAIN.BEGIN_EPOCH = 0
_C.TRAIN.END_EPOCH = 300

_C.TRAIN.EARLY_STOP = 30
_C.TRAIN.CHECKPOINT = ''

_C.TRAIN.BATCH_SIZE_PER_GPU = 32
_C.TRAIN.SHUFFLE = True



def update_config(cfg, args):
    cfg.defrost()
    cfg.merge_from_file(args.cfg)
    cfg.freeze()


if __name__ == '__main__':
    import sys
    with open(sys.argv[1], 'w') as f:
        print(_C, file=f)

def get_config():
    parser = argparse.ArgumentParser()
    # cleansing
    parser.add_argument('--path_tar', type=str, default='../autodl-tmp/dataset_ROP',
                        help='Path to the target folder to store the processed datasets.')
    parser.add_argument('--json_file_dict', type=str, default="./json_src",
                        help='Path to the source folder containing original datasets.')
    parser.add_argument('--generate_ridge', type=bool, default=False,
                        help='if generate the ridge cooridinate from json src.')
    # Cleansing
    parser.add_argument('--patch_size', type=int, default=256,
                        help='patch size in cleansing .')
    parser.add_argument('--stride', type=int, default=64,
                        help='stride in cleansing.')
    
    # Model
    parser.add_argument('--model', type=str, default='FR_UNet',
                        help='Name of the model architecture to be used for training.')
    
    # train and test
    parser.add_argument('--dataset', type=str, default="all",
                        help='Datset used. DRIONS-DB,GY,HRF,ODVOC,STARE | all')
    parser.add_argument('--save_name', type=str, default="./checkpoints/best.pth",
                        help='Name of the file to save the best model during training.')
    parser.add_argument('--result_path', type=str, default="experiments",
                        help='Path to the visualize result or the pytorch model will be saved.')
    parser.add_argument('--from_checkpoint', type=str, default="",
                        help='load the exit checkpoint.')
    
    # config file 
    parser.add_argument('--cfg', help='experiment configuration filename',
                        default="./YAML/default.yaml", type=str)
    
    args = parser.parse_args()
    # Merge args and config file 
    update_config(_C, args)
    args.configs=_C

    return args