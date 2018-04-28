# Create binarized data (I have to do sauvola binarization on my machine because we don't have it installed on the AWS instance).
#mkdir ~/datasets/read_ICFHR/test_data_howe
#mkdir ~/datasets/read_ICFHR/test_data_simplebin
#python batch_apply_binarization.py ~/datasets/read_ICFHR/test_data ~/datasets/read_ICFHR/test_data_howe 10 --howe --icfhr 
#python batch_apply_binarization.py ~/datasets/read_ICFHR/test_data ~/datasets/read_ICFHR/test_data_simplebin 10 --sauvola --icfhr 

# For russell's computer
# python batch_apply_binarization.py /deep_data/datasets/ICFHR_Data/test_data /deep_data/datasets/ICFHR_Data/test_data/test_data_simplebin 10 --sauvola --icfhr 

# Create lmdb database (REQUIRES ME TO MODIFY DEALING WITH TEXT FILES NOT PRESENT

# Oops, I will need an lmdb database for each test set, as I did before!

python create_dataset.py --data_dir ~/datasets/read_ICFHR/test_data --output_dir ~/russell/nephi/data/lmdb_ICFHR_bin/test_data --icfhr --binarize --howe_dir ~/datasets/read_ICFHR/test_data_howe --simplebin_dir ~/datasets/read_ICFHR/test_data_simplebin --test

# Now I need to determine all the models that did the best. I should do this in some automatic way, but I will do manually for now, 1 by 1

#python create_dataset.py  /deep_data/datasets/ICFHR_Data/general_data  /deep_data/nephi/data/lmdb_ICFHR/test_data/30865_testtrack --icfhr /deep_data/nephi/test.lst
#python create_dataset.py  /deep_data/datasets/ICFHR_Data/specific_data  /deep_data/nephi/data/lmdb_ICFHR/specific_data --icfhr

# Oops, I will need an lmdb database for each test set, as I did before!

Test loss:
Character error 
Saving epoch
grep "Test loss:\|Character error\|Saving epoch"

# I could possibly do a validation scheme where I train on the 12 pages and test on the 4 pages. Maybe get a better idea of where the epochs go for closer to 16 pages. This is probably needed because 12 pages is a ton more data than 4 pages. But an epoch is an epoch with stochastic gradient descent with same batch size, it has the chance to go over the data similarly.

30866_train_1  # This one could have actually trained longer to possible get better character error rate
experiments/expr_ICFHR_18Apr_finetuning_allnet_30866_train_1/netCRNN_14_5.pth
30866_train_4    # Maybe could have trained a little more but seemed to converge
experiments/expr_ICFHR_18Apr_finetuning_allnet_30866_train_4/netCRNN_13_20.pth
30866_train_16   # The last two epochs seemed to do worse on the training set for some random reason
Saving epoch experiments/expr_ICFHR_18Apr_finetuning_allnet_30866_train_16/netCRNN_12_79.pth
# Test loss clearly minimized here before fluctuating up
Saving epoch experiments/expr_ICFHR_18Apr_finetuning_allnet_30882_train_1/netCRNN_6_4.pth 
# same here
Saving epoch experiments/expr_ICFHR_18Apr_finetuning_allnet_30882_train_4/netCRNN_6_14.pth
# I am just going to choose one more epoch for the 16 page
Saving epoch experiments/expr_ICFHR_18Apr_finetuning_allnet_30882_train_16/netCRNN_7_55.pth
# Here it randomly minimized (after a lot of early bumpiness)
Saving epoch experiments/expr_ICFHR_18Apr_finetuning_allnet_30893_train_1/netCRNN_7_4.pth
# Character error rate, but not loss minimized here. The fine-tuning for this dataset is VERY chaotic, though the handwriting is super neat
Saving epoch experiments/expr_ICFHR_18Apr_finetuning_allnet_30893_train_4/netCRNN_12_15.pth
# It doesn't make sense to me that the 8 list really trailed the 16 list here. I just realized that the 8 list should really be a 12 list (12 pages).
Saving epoch experiments/expr_ICFHR_18Apr_finetuning_allnet_30893_train_16/netCRNN_14_64.pth
Saving epoch experiments/expr_ICFHR_18Apr_finetuning_allnet_35013_train_1/netCRNN_5_7.pth
Saving epoch experiments/expr_ICFHR_18Apr_finetuning_allnet_35013_train_4/netCRNN_6_26.pth
Saving epoch experiments/expr_ICFHR_18Apr_finetuning_allnet_35013_train_16/netCRNN_8_107.pth
# This could have possible used more training time.
Saving epoch experiments/expr_ICFHR_18Apr_finetuning_allnet_35015_train_1/netCRNN_14_12.pth
# maybe more training time, but fluctuated at end
Saving epoch experiments/expr_ICFHR_18Apr_finetuning_allnet_35015_train_4/netCRNN_7_44.pth
Saving epoch experiments/expr_ICFHR_18Apr_finetuning_allnet_35015_train_16/netCRNN_8_177.pth





