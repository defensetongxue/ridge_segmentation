
# # python generate_mask.py
python -u train.py --split_name clr_1
python -u test.py --split_name clr_1
python -u train.py  --split_name clr_2
python -u test.py --split_name clr_2
python -u train.py  --split_name clr_3
python -u train.py  --split_name clr_3
# python -u test.py --split_name clr_4
