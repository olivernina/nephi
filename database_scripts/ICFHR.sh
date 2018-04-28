#python create_dataset.py  ~/datasets/read_ICFHR/general_data ~/russell/nephi/data/lmdb_ICFHR/general_data --icfhr


python create_dataset.py  --data_dir ~/datasets/read_ICFHR/general_data  --output_dir ~/russell/nephi/data/lmdb_ICFHR_bin/general_data --icfhr --binarize --howe_dir ~/datasets/read_ICFHR/general_data_howe --simplebin_dir ~/datasets/read_ICFHR/general_data_imgtxt
python create_dataset.py  --data_dir ~/datasets/read_ICFHR/specific_data --output_dir ~/russell/nephi/data/lmdb_ICFHR_bin/specific_data --icfhr --binarize --howe_dir ~/datasets/read_ICFHR/specific_data_howe --simplebin_dir ~/datasets/read_ICFHR/specific_data_imgtxt
