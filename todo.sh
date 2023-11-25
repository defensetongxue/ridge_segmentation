python cleansing.py
python -u train.py 
python segment_test.py 
cd ../SentenceROP
python cleansing.py
python -u train.py --aux_r 0.0
python -u train.py --aux_r 1.0
python ring.py
shutdown