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
hex_ = hex_[0:-1:2]


def plot(column, metric, smoothing, work_dir):

    # pretty_colors = ['#FC474C','#8DE047','#FFDD50','#53A3D7']

    max_x = 0
    max_y = 0

    column_num = column #cider_val = -8, blue4_val=11, ..., ROUGE= 10,METEOR=11

    # files = os.listdir(work_dir)
    # dirs = []
    # res_files = [os.path.join(work_dir,file,'train_valid_test.txt') for file in files if os.path.exists(os.path.join(work_dir,file,'train_valid_test.txt'))]

    res_files = [os.path.join(work_dir,'plot.txt')]
    init_val = 0

    max_y = -9999
    max_x = -9999
    data_x_y_enum_name = []

    # Do one pass to get max value
    for i, filename in enumerate(sorted(res_files)):
        # if filename.split('/')[-2].endswith('iters-4') or filename.split('/')[-2].endswith('iters-12'):
        #     continue

        data = genfromtxt(filename, delimiter=' ')
        # if len(data) == 22:
        #     continue
        x = data[init_val:, 0]
        y = data[init_val:, column_num]

        if smoothing:
            x_smooth = linspace(x.min(), x.max(), 1000)
            akima = Akima1DInterpolator(x, y)
            y_smooth = akima(x_smooth)
            x = x_smooth
            y = y_smooth

        if x.max() > max_x:
            max_x = x.max()
        if y.max() > max_y:
            max_y = y.max()

        # data_x_y_enum_name.append((x, y, i, filename.split('/')[-2]))
        data_x_y_enum_name.append((x, y, i, 'CRNN'))

    fig = pyplot.figure(figsize=(6, 6))
    axes = pyplot.gca()
    pyplot.grid()

    bufferx = 0.25 * max_x
    buffery = 0.25 * max_y
    axes.set_ylim([0, max_y + buffery])
    # axes.set_ylim([0,0.01])
    axes.set_xlim([1, max_x + bufferx])
    # axes.set_xlim([0, 100])
    pyplot.xlabel('Iterations')
    pyplot.ylabel('{}'.format(metric.upper()))
    pyplot.title(metric)

    for x, y, enum, name in data_x_y_enum_name:
        # Will crash if file only has 1 line.
        try:
            pyplot.plot(x, y, linewidth=2, label=name, color=hex_[enum])
        except IndexError as e:
            print("EXCEPTION: " + e.message)
            print 'Failed to create plot for {}.\nIs there only 1 epoch?'.format(name)
            continue

    pyplot.legend(loc='lower right', shadow=True, fontsize='medium')
    pyplot.savefig(os.path.join(work_dir, '{}.eps'.format(metric)))
    pyplot.savefig(os.path.join(work_dir, '{}.png'.format(metric)))
    print "Plotted {} series".format(len(data_x_y_enum_name))



if __name__=="__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('-w', dest='work_dir', type=str)
    arg_parser.add_argument('-p', dest='plot_type', type=str)
    arg_parser.add_argument('-s', '--smotthing', dest='smoothing', type=int, default=0)

    if not len(sys.argv) > 1:
        arg_parser.print_help()
        sys.exit(0)

    args = arg_parser.parse_args()
    plot_type = args.plot_type
    smoothing = args.smoothing
    work_dir = args.work_dir

    if plot_type == 'loss':
        plot(-4, 'loss', smoothing, work_dir)

    elif plot_type == 'werr':
        plot(-3, 'werr', smoothing, work_dir)

    elif plot_type == 'cerr':
        plot(-2, 'cerr', smoothing, work_dir)

    elif plot_type == 'accu':
        plot(-1, 'accu', smoothing, work_dir)

    elif plot_type == 'all':
        plot(-4, 'loss', smoothing, work_dir)
        plot(-3, 'werr', smoothing, work_dir)
        plot(-2, 'cerr', smoothing, work_dir)
        plot(-1, 'accu', smoothing, work_dir)
    else:
    	print plot_type+" metric not supported"
