python crnn_main.py --trainroot data/lmdb/train --valroot data/lmdb/val --cuda --lr 0.0001 --displayInterval 1 --valEpoch 1 --saveEpoch 1 --workers 10 --niter 100 --experiment expr_test_worderror > log_test_worderror.txt

