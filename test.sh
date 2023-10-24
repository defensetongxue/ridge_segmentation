python -u test.py --cfg ./config_file/hrnet_small_finetone.json
python  -u cleansing.py --stride 64
python -u train.py --cfg ./config_file/hrnet_small.json