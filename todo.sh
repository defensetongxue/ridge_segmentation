python -u train.py --cfg ./config_file/unet.json --save_name unet.bth
python -u test.py  --cfg ./config_file/unet.json --save_name unet.bth
python -u train.py --cfg ./config_file/wnet.json --save_name wnet.bth
python -u test.py --cfg ./config_file/wnet.json --save_name wnet.bth
python -u train.py --cfg ./config_file/hrnet.json --save_name wnet.bth
python -u test.py --cfg ./config_file/hrnet.json --save_name wnet.bth
python -u train.py --cfg ./config_file/unet1.json --save_name unet1.bth
python -u test.py --cfg ./config_file/unet1.json --save_name unet1.bth
python -u train.py --cfg ./config_file/wnet1.json --save_name wnet1.bth
python -u test.py --cfg ./config_file/wnet1.json --save_name wnet1.bth
python -u train.py --cfg ./config_file/hrnet1.json --save_name hrnet1.bth
python -u test.py --cfg ./config_file/hrnet1.json --save_name hrnet1.bth
python ring.py