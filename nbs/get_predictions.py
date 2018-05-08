__author__ = 'onina'

# execute this script from the home directory i.e. python nps/finetune.py 0 0 

import os
import sys
from time import gmtime, strftime
import os
import sys
from glob import glob
import shutil
import subprocess

def find_models(work_dir,model):


    files = os.listdir(work_dir)
    dirs = [os.path.join(work_dir,file) for file in files if os.path.isdir(os.path.join(work_dir,file))]

    res_files = []
    for dir in dirs:
        files = os.listdir(dir)
        for file in files:
            if file == model:
                plt_path = os.path.join(dir, file, 'plot.txt')
                if os.path.exists(plt_path):
                    res_files.append(plt_path)

    column_num = -1 # CER test

    best_models = []
    for i, filename in enumerate(sorted(res_files)):

        data = genfromtxt(filename, delimiter=' ')

        CERs = data[:, column_num]

        min_idx = CERs.argmin()
        epoch = int(data[min_idx, 0])
        i = int(data[min_idx, 1])
        best_model_name = 'netCNN_'+str(epoch)+'_'+str(i)+'.pth'

        bm_path = os.path.join(filename.split('plot.txt')[0],best_model_name)
        if os.path.exists(bm_path):
            best_models.append(bm_path)
            print(bm_path)
        else:
            print('Model file not found '+bm_path)
            sys.exit(0)

    return best_models

def create_qsub_file(name, bash_command):
    filename =  "pbs/icfhr/test/"+name + '.pbs'
    f = open(filename, 'w')
    f.write("#!/bin/bash \n")
    f.write("#PBS -A AFSNW35489ANO\n")
    f.write("#PBS -l select=1:ncpus=28:ompthreads=8:ngpus=2\n")
    f.write("#PBS -q GPU_RD\n")
    f.write("#PBS -l walltime=05:00:00\n")
    f.write("#PBS -N " + name + "\n")
    f.write("#PBS -j oe\n")
    f.write("source /home/oliver/.personal.bashrc\n")
    f.write("cd /p/home/oliver/projects/nephi\n")
    f.write(bash_command)
    f.write("exit\n")
    f.close()


    return filename

def submit_qsub(filename):
    os.system("cd pbs/icfhr/test")
    command = 'qsub ' + filename
    print(command)
    print(filename)
    os.system(command)

def main(argv):

    DEBUG = int(argv[1])
    res_dir = argv[2]
    model = argv[3]


    lmdb_database_base = "data/lmdb_ICFHR_bin/specific_data_each_doc/"
    spec_tr_lists_dir = "data/datasets/read_ICFHR/specific_data_train_list/"

    spec_lists_files = glob(os.path.join(spec_tr_lists_dir, "*"))
    
    dirs = [os.path.basename(f).partition(".lst")[0] for f in spec_lists_files]

    lmdb_database_base = "data/lmdb_ICFHR_bin/specific_data_each_doc/"
    # pre_model = "results_icfhr_aug/attention+ctc/netCNN_24_2982.pth"

    tuned_models = find_models(res_dir,model)

    job_id = 0
    task_idx=0
    for num in set([d.partition("_train_")[0] for d in dirs]):
        for s in ["0", "1", "16", "4"]:
            if s == "0":
                bash_command = ' '.join(["python crnn_main.py", "--trainroot", os.path.join(lmdb_database_base, num),
                                   "--valroot", os.path.join(lmdb_database_base, num),
                                   "--dataset ICFHR --cuda --lr 0.0001 --displayInterval 4 --valEpoch 1 --saveEpoch 1 --workers 3",
                                   "--niter 60 --keep_ratio --imgH 60 --imgW 240 --batchSize 4",
                                   "--transform --rescale --rescale_dim 3 --grid_distort",
                                   "--pre_model",
                                   "best_models/"+model+"/netCNN_24_2982.pth",
                                   "--test_icfhr --test_file",
                                   os.path.join("test_results/7May_firstfinetune_submission", num + "_" + s + ".txt"),
                                   ">",
                                   "logs/test/"+model +"/log_7May_test_results_" + num + "_" + s + ".txt"])

            else:
                bash_command = ' '.join(["python crnn_main.py", "--trainroot", os.path.join(lmdb_database_base, num),
                                   "--valroot", os.path.join(lmdb_database_base, num),
                                   "--dataset ICFHR --cuda --lr 0.0001 --displayInterval 4 --valEpoch 1 --saveEpoch 1 --workers 3",
                                   "--niter 60 --keep_ratio --imgH 60 --imgW 240 --batchSize 4",
                                   "--transform --rescale --rescale_dim 3 --grid_distort",
                                   "--pre_model", tuned_models[task_idx],
                                   "--test_icfhr --test_file",
                                   os.path.join("test_results/7May_firstfinetune_submission", num + "_" + s + ".txt"),
                                   ">",
                                    "logs/test/"+model +"/log_7May_test_results_" + num + "_" + s + ".txt"])
                task_idx+=1



            if DEBUG:
                print(bash_command)
                filename = create_qsub_file(str(job_id), bash_command)
                sys.exit(0)

            else:
                filename = create_qsub_file(str(job_id),bash_command)
                submit_qsub(filename)

            job_id+=1

if __name__=="__main__":


   main(sys.argv)
