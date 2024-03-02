
# # python generate_mask.py
python -u train_vit.py  --split_name clr_1 --cfg ./config_file/transunet.json --lr 1e-4 --wd 5e-4
python -u train_vit.py  --split_name clr_2 --cfg ./config_file/transunet.json --lr 1e-4 --wd 5e-4
python -u train_vit.py  --split_name clr_4 --cfg ./config_file/transunet.json --lr 1e-4 --wd 5e-4
python -u train_vit.py  --split_name clr_4 --cfg ./config_file/transunet.json --lr 1e-4 --wd 5e-4
python -u test_vit.py  --split_name clr_1 --cfg ./config_file/transunet.json --lr 1e-4 --wd 5e-4
python -u test_vit.py  --split_name clr_2 --cfg ./config_file/transunet.json --lr 1e-4 --wd 5e-4
python -u test_vit.py  --split_name clr_4 --cfg ./config_file/transunet.json --lr 1e-4 --wd 5e-4
python -u test_vit.py  --split_name clr_4 --cfg ./config_file/transunet.json --lr 1e-4 --wd 5e-4
python -u test_sz_vit.py  --split_name clr_1 --cfg ./config_file/transunet.json --lr 1e-4 --wd 5e-4