python cleansing.py
python -u train.py 
python segment_test.py 
cd ../SentenceROP
python cleansing.py
python -u train.py --lr 5e-4 --wd 5e-4
python -u train.py --lr 1e-3 --wd 5e-4
python -u train.py --lr 5e-4 --wd 5e-3
python -u train.py --lr 5e-4 --wd 5e-2
python ring.py
shutdown