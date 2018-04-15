import howe

from os import listdir, makedirs, system
from os.path import isfile, join, exists, splitext, basename

import multiprocessing
import itertools
import time
import cv2
from glob import glob

import sys
images_folder = sys.argv[1]
output_folder = sys.argv[2]

processes = int(sys.argv[3])

binarizer = sys.argv[4]

if binarizer not in ["--howe", "--sauvola"]:
    print("ERROR: No or wrong binarizer specified")
    exit(-1)

ICFHR = False

if (len(sys.argv) == 6) and (sys.argv[5] == "--icfhr"):
    ICFHR = True
    
print processes

if not exists(output_folder):
    makedirs(output_folder)

image_files = None
if ICFHR:
    image_files = glob(join(images_folder, "*/*/*.jpg"))
else:
    image_files = [f for f in listdir(images_folder) if isfile(join(images_folder, f)) and ".jpg" in f.lower()]

def grouper(n, iterable):
    it = iter(iterable)
    while True:
       chunk = tuple(itertools.islice(it, n))
       if not chunk:
           return
       yield chunk

def process_image(params):
    start = time.time()
    filename, input_path, output_path = params
    bin_name = "howe" if binarizer == "--howe" else "simplebin"
    
    if ICFHR:
        output_file = join(output_path, basename(filename).lower().partition(".jpg")[0] + "_" + bin_name + ".jpg")
    else:
        output_file = join(output_path, filename.lower().partition(".jpg")[0] + "_" + bin_name + ".jpg")
    #print filename
    if exists(output_file):
        return
    image = cv2.imread(join(input_path, filename))
    if binarizer == "--howe":
        result = howe.binarize(image)
        cv2.imwrite(output_file, result)
    else:
        run_str = "/deep_data/imgtxtenh/build/imgtxtenh -w 30 -d 7.0 -s 0.5 -S 1 {} {}"
        run_str = run_str.format(join(input_path, filename), output_file)
        print run_str
        ret = system(run_str)
        if ret != 0:
            raise Exception("BREAK ERROR")
    print ("Saved Howe binarization from %s to %s in time: %s" % (join(input_path, filename), output_file, str(time.time() - start)))


    
#print image_files
pool = multiprocessing.Pool(processes=processes)
# for i, image_data in enumerate(grouper(processes, image_files)):
#     print i * processes, " images processed"
#     func_params = [(v, images_folder, output_folder) for v in image_data]
#     result = pool.map(process_image, func_params)

func_params = [(v, images_folder, output_folder) for v in image_files]
result = pool.map(process_image, func_params)
