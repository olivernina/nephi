__author__ = 'onina'

import os
import sys
from time import gmtime, strftime
import os
import sys
from glob import glob
import shutil
import subprocess

def main(argv):
    # lmdb_database_base = "/deep_data/nephi/data/lmdb_ICFHR/specific_data_each_doc"
    # spec_tr_lists_dir = "/deep_data/datasets/ICFHR_Data/specific_data_train_list"

    lmdb_database_base = "/home/ubuntu/russell/nephi/data/lmdb_ICFHR_bin/specific_data_each_doc/"
    spec_tr_lists_dir = "/home/ubuntu/datasets/read_ICFHR/specific_data_train_list/"

    spec_lists_files = glob(os.path.join(spec_tr_lists_dir, "*"))
    dirs = [os.path.basename(f).partition(".lst")[0] for f in spec_lists_files]

    lmdb_database_base = "/home/ubuntu/russell/nephi/data/lmdb_ICFHR_bin/specific_data_each_doc/"
    pre_model = "/home/ubuntu/russell/nephi/trained_models/results_icfhr_aug/attention+ctc/netCNN_24_2982.pth"



    for num in set([d.partition("_train_")[0] for d in dirs]):
        for s in ["1", "4", "16"]:
            command = ' '.join(["python crnn_main.py", "--trainroot", os.path.join(lmdb_database_base, num + "_train_" + s),
                               "--valroot", os.path.join(lmdb_database_base, num + "_train_" + "8"),
                               "--dataset ICFHR --cuda --lr 0.0001 --displayInterval 4 --valEpoch 1 --saveEpoch 1 --workers 3",
                               "--niter 60 --keep_ratio --imgH 60 --imgW 240 --batchSize 4",
                               "--transform --rescale --rescale_dim 3 --grid_distort",
                               "--model attention+ctc --plot",
                               "--rdir", "experiments/7May_finetuning/expr_" + "ICFHR_7May_finetuning_attention+ctc_" + num + "_train_" + s,
                               "--pre_model",  pre_model, ">",
                               "log_files/log_ICFHR_7May_finetuning_attention+ctc_" + num + "_train_" + s + ".txt"])
            print(command)

            # os.system(command)

if __name__=="__main__":


   # if len(sys.argv)<2:
   #     print "Need more arguments \ncluster_jobs.py start end step"
   # else:
   main(sys.argv)
