python -u train.py --cfg ./config_file/hrnet_no.json --save_name henet_nopos.bth
python -u test.py  --cfg ./config_file/hrnet_no.json --save_name henet_nopos.bth
python ring.py