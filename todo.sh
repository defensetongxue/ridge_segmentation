python  -u cleansing.py --split_name clr_1
python -u train.py --split_name clr_1
python -u dynamic_r_test.py --split_name clr_1

python  -u cleansing.py --split_name clr_2
python -u train.py --split_name clr_2
python -u dynamic_r_test.py --split_name clr_2

python  -u cleansing.py --split_name clr_3
python -u train.py --split_name clr_3
python -u dynamic_r_test.py --split_name clr_3

python  -u cleansing.py --split_name clr_4
python -u train.py --split_name clr_4
python -u dynamic_r_test.py --split_name clr_4
shutdown