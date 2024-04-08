
# # python generate_mask.py
python -u test.py --split_name clr_1 
python -u test.py --split_name clr_2 
python -u test.py --split_name clr_3 
python -u test.py --split_name clr_4 
python -u test_vit.py --split_name clr_1 
python -u test_vit.py --split_name clr_2
python -u test_vit.py --split_name clr_3 
python -u test_vit.py --split_name clr_4 
python -u test.py --split_name clr_1 --config ./config_file/unet_torch.json
python -u test.py --split_name clr_2 --config ./config_file/unet_torch.json
python -u test.py --split_name clr_3 --config ./config_file/unet_torch.json
python -u test.py --split_name clr_4 --config ./config_file/unet_torch.json

python train.py --split_name all --config ./config_file/unet_torch.json