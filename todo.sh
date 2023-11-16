python -u train.py --split_name 3
python -u train.py --cfg ./config_file/hrnet_small.json --split_name 4
python -u checkfrom_map.py 
python ring.py
shutdown
# python -u train.py --cfg ./config_file/hrnet_small_all.json 
# python -u test.py --cfg ./config_file/hrnet_small_all.json 
# python -u train.py --cfg ./config_file/hrnet_small_all1.json 
# python -u test.py --cfg ./config_file/hrnet_small_all1.json 
# python -u train.py --cfg ./config_file/hrnet_small_finetone.json
# python -u test.py --cfg ./config_file/hrnet_small_finetone.json