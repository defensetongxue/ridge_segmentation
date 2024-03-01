
# # python generate_mask.py
python -u test.py --split_name clr_1 --cfg ./config_file/hrnet_w48.json   
# python -u test.py --split_name clr_2 --cfg ./config_file/hrnet_w48.json   
# python -u test.py --split_name clr_3 --cfg ./config_file/hrnet_w48.json   
# python -u test.py --split_name clr_4 --cfg ./config_file/hrnet_w48.json   
python -u test.py --split_name clr_1 --cfg ./config_file/unet_small.json   
# python -u test.py --split_name clr_2 --cfg ./config_file/unet_small.json   
# python -u test.py --split_name clr_3 --cfg ./config_file/unet_small.json   
# python -u test.py --split_name clr_4 --cfg ./config_file/unet_small.json   
python -u test_vit.py  --split_name clr_1 --cfg ./config_file/transunet.json
# python -u test_vit.py  --split_name clr_2 --cfg ./config_file/transunet.json
# python -u test_vit.py  --split_name clr_3 --cfg ./config_file/transunet.json
# python -u test_vit.py  --split_name clr_4 --cfg ./config_file/transunet.json
cd ../ROP_BaseLine/
sh todo.sh