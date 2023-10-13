python -u train.py --cfg ./config_file/hrnet_w48.json --save_name hrnet_large.bth
python -u test.py  --cfg ./config_file/hrnet_w48.json --save_name hrnet_lage.bth
python -u train.py --cfg ./config_file/unet_torch.json --save_name unet_torch.bth
python -u test.py  --cfg ./config_file/unet_torch.json --save_name unet_torch.bth
python ring.py