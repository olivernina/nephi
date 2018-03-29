
# Modified the script to continue training where there was an error.
# python crnn_main.py --trainroot /deep_data/nephi/data/lmdb_ICFHR/general_data --valroot /deep_data/nephi/data/lmdb_ICFHR/specific_data --crnn  /deep_data/nephi/experiments/expr_ICFHR_27Mar_alph_werr_fixed/netCRNN_0_5963.pth --cuda --lr 0.0001 --displayInterval 120 --valEpoch 5 --saveEpoch 5 --workers 10 --niter 200 --experiment experiments/expr_ICFHR_27Mar_alph_werr_fixed --keep_ratio --imgH 80 --imgW 240 --batchSize 2 > log_ICFHR_27Mar_alph_werr_fixed.txt
# On 20 march 2018 I lowered the lr from 0.0001 to 0.00005 and increased the batch size from 4 to 8. I changed names from "extend" to "extended". The problem I have been having is a memory allocation error when loading the line images of the training set for validation. I should probably change memory here somehow, but the temporary fix is that I've decreased the number of batches to load.

# On 28 March 2018, this code now works pretty well. Batchsize 2 worked on my computer. No errors.

# During the training on 28 March 2018, somehow the training got interrupted (this time not from memory allocation, maybe it was to reduce the temperature of the machine). I picked up where it left off (epoch 15, though this was 15 ahead of epoch 5), turning the learning rate down from 0.0001 to 0.00001.

python crnn_main.py --trainroot /deep_data/nephi/data/lmdb_ICFHR/general_data --valroot /deep_data/nephi/data/lmdb_ICFHR/specific_data --crnn  /deep_data/nephi/experiments/expr_ICFHR_27Mar_alph_werr_fixed/netCRNN_15_5963.pth --cuda --lr 0.00001 --displayInterval 120 --valEpoch 5 --saveEpoch 5 --workers 10 --niter 200 --experiment experiments/expr_ICFHR_27Mar_alph_werr_fixed_extended --keep_ratio --imgH 80 --imgW 240 --batchSize 2 >> log_ICFHR_27Mar_alph_werr_fixed.txt
