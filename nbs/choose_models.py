__author__ = 'oliver'

from numpy import genfromtxt
import argparse
import os
import sys


def find_models(work_dir):


    files = os.listdir(work_dir)
    dirs = [os.path.join(work_dir,file) for file in files if os.path.isdir(os.path.join(work_dir,file))]

    res_files = []
    for dir in dirs:
        files = os.listdir(dir)
        for file in files:
            plt_path = os.path.join(dir, file, 'plot.txt')
            if os.path.exists(plt_path):
                res_files.append(plt_path)

    column_num = -1 # CER test

    # Do one pass to get max value
    for i, filename in enumerate(sorted(res_files)):

        data = genfromtxt(filename, delimiter=' ')

        CERs = data[:, column_num]

        min_idx = CERs.argmin()
        epoch = int(data[min_idx, 0])
        i = int(data[min_idx, 1])
        best_model_name = 'netCNN_'+str(epoch)+'_'+str(i)+'.pth'

        bm_path = os.path.join(filename.split('plot.txt')[0],best_model_name)
        if os.path.exists(bm_path):
            print(bm_path)
        else:
            print('Model file not found '+bm_path)
            sys.exit(0)

if __name__=="__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('-w', dest='work_dir', type=str)

    if not len(sys.argv) > 1:
        arg_parser.print_help()
        sys.exit(0)

    args = arg_parser.parse_args()

    work_dir = args.work_dir


    find_models( work_dir)



