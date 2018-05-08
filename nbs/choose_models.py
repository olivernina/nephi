__author__ = 'oliver'


import matplotlib
# matplotlib.use('Agg') # Or any other X11 back-end


import matplotlib.pyplot as pyplot
from matplotlib import colors
import argparse

import sys
from numpy import genfromtxt, linspace
from scipy.interpolate import Akima1DInterpolator
import os
import six

xmin = 20000

colors_ = list(six.iteritems(colors.cnames))

# Add the single letter colors.
for name, rgb in six.iteritems(colors.ColorConverter.colors):
    hex_ = colors.rgb2hex(rgb)
    colors_.append((name, hex_))

# Transform to hex color values.
hex_ = [color[1] for color in colors_]
# shuffle(hex_)
hex_ = hex_[2:-1:2]


def find_models(work_dir):


    files = os.listdir(work_dir)
    res_files = [os.path.join(work_dir,file,'plot.txt') for file in files if os.path.exists(os.path.join(work_dir,file,'plot.txt'))]

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



