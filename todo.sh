python -u train.py --cfg ./config_file/hrnet_small.json
python -u test.py --cfg ./config_file/hrnet_small.json
python -u finetone.py --cfg  ./config_file/hrnet_small_finetone.json
python -u test.py --cfg  ./config_file/hrnet_small_finetone.json