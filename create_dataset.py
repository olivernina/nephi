import os
import lmdb # install lmdb by "pip install lmdb"
import cv2
import numpy as np
from tool.xml_parser import page_images
from glob import glob
import re
import sys
import io
import argparse
from scipy.spatial import distance
encoding = 'utf-8'

stdout = sys.stdout
reload(sys)  
sys.setdefaultencoding('utf-8')
sys.stdout = stdout

def checkImageIsValid(imageBin):
    if imageBin is None:
        return False
    imageBuf = np.fromstring(imageBin, dtype=np.uint8)
    img = cv2.imdecode(imageBuf, cv2.IMREAD_GRAYSCALE)
    imgH, imgW = img.shape[0], img.shape[1]
    if imgH * imgW == 0:
        return False
    return True


# basically "flush the cache to the actual DB"
def writeCache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.iteritems():
            txn.put(k, v)


def createDataset(outputPath, imagePathList, labelList, lexiconList=None, checkValid=True):
    """
    Create LMDB dataset for CRNN training.
    ARGS:
        outputPath    : LMDB output path
        imagePathList : list of image path
        labelList     : list of corresponding groundtruth texts
        lexiconList   : (optional) list of lexicon lists
        checkValid    : if true, check the validity of every image
    """
    assert(len(imagePathList) == len(labelList))
    nSamples = len(imagePathList)
    env = lmdb.open(outputPath, map_size=1099511627776)
    cache = {}
    cnt = 1
    for i in xrange(nSamples):
        imagePath = imagePathList[i]
        print imagePath
        label = labelList[i]
        if not os.path.exists(imagePath):
            print('%s does not exist' % imagePath)
            continue
        with open(imagePath, 'r') as f:
            imageBin = f.read()
        if checkValid:
            if not checkImageIsValid(imageBin):
                print('%s is not a valid image' % imagePath)
                continue

        imageKey = 'image-%09d' % cnt
        labelKey = 'label-%09d' % cnt
        cache[imageKey] = imageBin
        cache[labelKey] = label
        if lexiconList:
            lexiconKey = 'lexicon-%09d' % cnt
            cache[lexiconKey] = ' '.join(lexiconList[i])
        if cnt % 1000 == 0:
            writeCache(env, cache)
            cache = {}
            print('Written %d / %d' % (cnt, nSamples))
        cnt += 1
    nSamples = cnt-1
    cache['num-samples'] = str(nSamples)
    writeCache(env, cache)
    print('Created dataset with %d samples' % nSamples)

# From: https://stackoverflow.com/questions/24467972/calculate-area-of-polygon-given-x-y-coordinates
#def PolyArea(x,y):
#    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))

def PolyArea(x,y):
    correction = x[-1] * y[0] - y[-1]* x[0]
    main_area = np.dot(x[:-1], y[1:]) - np.dot(y[:-1], x[1:])
    return 0.5*np.abs(main_area + correction)

# Takes an image read by cv2 and masks out the region of interest (pts)
def apply_mask(img, pts, add_pixel = False):
    
    pts = np.array(pts, np.int32)

    xmin = min(pts, key=lambda x: x[0])[0]
    xmax = max(pts, key=lambda x: x[0])[0]

    ymin = min(pts, key=lambda x: x[1])[1]
    ymax = max(pts, key=lambda x: x[1])[1]
    
    #if False:
    if add_pixel:
        ymin = ymin - add_pixel
        if ymin < 0:
            ymin = 0
        print("Ymin:")
        print(ymin)
        ymax = ymax + add_pixel
        if ymax >= img.shape[0]:
            ymax = img.shape[0] - 1
        
        print("Ymax")
        print(ymax)
        print("IMage shape:")
        print(img.shape)
        
    # RA: I will probably have to make allowance for the inevitable error that for a first or last line on the page, adding pixels takes us off the page.
    
    # RA: I am now just going to use the whole array, given that they are ordered correctly
    updated_pts = np.array([(p[0] - xmin, p[1] - ymin) for p in pts], np.int32)
    #if False:
    #if isinstance(add_pixel, (int, long)):
    if add_pixel:
        #x_pts = np.expand_dims(np.array([x[0] for x in updated_pts]), axis=1)
        #print("Shape and dimensions of x_pts")
        #print(x_pts.shape)
        #print(x_pts.ndim)
        #d_array = distance.cdist(x_pts, x_pts, 'euclidean')   # only care about x-distance
        
        for i, pt in enumerate(updated_pts):
            area_poly = PolyArea(updated_pts[:,0], updated_pts[:,1])
            up_pts = updated_pts.copy()
            down_pts = updated_pts.copy()
            up_pts[i,1] = up_pts[i,1] + add_pixel
            down_pts[i,1] = down_pts[i,1] - add_pixel
            
            if PolyArea(up_pts[:,0], up_pts[:,1]) > area_poly:
                updated_pts[i,1] = updated_pts[i,1] + add_pixel
            elif PolyArea(down_pts[:,0], down_pts[:,1]) > area_poly:
                updated_pts[i,1] = updated_pts[i,1] - add_pixel
            if updated_pts[i,1] < 0:
                updated_pts[i,1] = 0
            elif updated_pts[i,1] > ymax:
                updated_pts[i,1] = ymax
            
            # First closest point code below:
            
            # Find the 7 closest points along the x-axis
            #closest_x_pts = np.argpartition(d_array[:,i], 8)[:8]  # includes index of the first point
            #print("Indecies of closest_x_pts")
            #print(closest_x_pts)
            # k smallest elements
            #np.argpartition(arr, k)[:k]
            #closest_pts = pts[np.array(closest_x_pts)]
            #print("Current point considering")
            #print(pt)
            #print("Actual closes_x_pts")
            #print(closest_pts)
            
            # Find whether increasing pixel height or decreasing pixel height adds to the area of the region of interest
            #area_poly = PolyArea(closest_pts[:,0], closest_pts[:,1])
            #print("Area of polygon")
            #print(area_poly)
            #up_closest_pts = closest_pts.copy()
            #down_closest_pts = closest_pts.copy()
            #pt_idx = np.where(np.all(np.isin(closest_pts, pt), axis=1))[0][0]
            #print("Point index")
            #print(pt_idx)
            #up_closest_pts[pt_idx,1] = up_closest_pts[pt_idx,1] + add_pixel
            #down_closest_pts[pt_idx,1] = down_closest_pts[pt_idx,1] - add_pixel
            #if PolyArea(up_closest_pts[:,0], up_closest_pts[:,1]) > area_poly:
            #    updated_pts[i,1] = updated_pts[i,1] + add_pixel
            #elif PolyArea(down_closest_pts[:,0], down_closest_pts[:,1]) > area_poly:
            #    updated_pts[i,1] = updated_pts[i,1] - add_pixel
    
    line_img = img[ymin:ymax, xmin:xmax].copy()
    mask = np.zeros(line_img.shape, dtype=np.uint8)
    
    
    
    
    channel_count = 1
    if len(line_img.shape) > 2:
        channel_count = line_img.shape[2]

    ignore_mask_color = (255,) * channel_count
    
    # Idiosyncrasy of cv2.fillPoly
    updated_pts = [(p[0], p[1]) for p in updated_pts]
    roi_corners = np.array([updated_pts], dtype=np.int32)
    
    cv2.fillPoly(mask, roi_corners, ignore_mask_color)
    line_img[mask == 0] = 255
    
    return line_img


def simple_dataset_from_dir(image_dir, output_path): 
    # a simple example of generating data (does not generate an alphabet.txt file, generate your own out of band)
    # pass an image_dir like data/dataset/images/train that contains files like
    # 25_this is the contents.png
    imagePathList = []
    labelList = []
    files = os.listdir(image_dir)
    for file in files:
        image_path = file
        imagePathList.append(os.path.join(image_dir,image_path)) # full path
        label = os.path.splitext(file.split('_')[1])[0] # "victor" from 25_victor.png
        print(file, label)
        labelList.append(label)

    createDataset(output_path, imagePathList, labelList)

def russell_page_journal(data_dir, output_path):
    env = lmdb.open(output_path, map_size=1099511627776)
    cache = {}
    cnt = 1
    
    img_files = glob(os.path.join(data_dir, "*.jpg"))
    alpha_text = u'' #'0123456789abcdefghijklmnopqrstuvwxyz'
    alphabet = []
    for img_file in img_files:
        img_c = cv2.imread(img_file)
        text_file = img_file.partition(".")[0] + ".txt"
        t_f = io.open(text_file, "r", encoding=encoding)
        gt = t_f.read()
        t_f.close()
        line_img = img_c

        imageBin = cv2.imencode('.png', line_img)[1].tostring()

        if not checkImageIsValid(imageBin):
            print('%s is not a valid image' % img_file)
            continue


        annotation = gt
        label = annotation.encode('utf-8')
        
        print("Printing encoded unicode!")
        print(label)

        for c in annotation:
            if not c in alphabet:
                alphabet.append(c)

        imageKey = 'image-%09d' % cnt
        labelKey = 'label-%09d' % cnt
        fileKey = 'file-%09d' % cnt

        print imageKey

        cache[imageKey] = imageBin
        cache[labelKey] = label
        cache[fileKey] = os.path.basename(img_file)

        if cnt % 1000 == 0:
            writeCache(env, cache)
            cache = {}
            print('Written %d' % (cnt))

        cnt += 1

    nSamples = cnt - 1
    cache['num-samples'] = str(nSamples)
    writeCache(env, cache)
    print('Created dataset with %d samples' % nSamples)
    alpha_text = u''.join(alphabet)
    with io.open("alphabet.txt", "w", encoding=encoding) as text_file:
        text_file.write(alpha_text)

# read into LMDB dataset from ICFHR 2018
def icfhr_dataset_read(data_dir, output_path, include_files=None, binarize = False, howe_dir = False, simplebin_dir = False, test = False):

    env = lmdb.open(output_path, map_size=1099511627776)
    cache = {}
    cnt = 1
    
    img_files = glob(os.path.join(data_dir, "*/*.jpg")) if test else glob(os.path.join(data_dir, "*/*/*.jpg"))
    for img_file in img_files:
        img_c = cv2.imread(img_file)
        info_file = img_file + ".info"
        if include_files is not None:
            if ".jpg" not in include_files[0]:
                include_files = [f + ".jpg" for f in include_files]
            if os.path.basename(img_file) not in include_files:
                continue
        if not test:
            text_file = img_file + ".txt"
        if binarize:
            howe_img = cv2.imread(os.path.join(howe_dir, os.path.basename(img_file).lower().partition(".jpg")[0] + "_howe.jpg"))
            simplebin_img = cv2.imread(os.path.join(simplebin_dir, os.path.basename(img_file).lower().partition(".jpg")[0] + "_simplebin.jpg"))
        
        with open(info_file, "r") as i_f:
            if not test:
                t_f = io.open(text_file, "r", encoding=encoding)
                gt = t_f.read()
                t_f.close()
            
            info = i_f.read()           
            mask = info.partition("MASK\n")[2]

            myre = re.compile(r"[0-9]+,[0-9]+")
            mask_p = myre.findall(mask)
            mask_pts = [tuple(int(x) for x in v.split(',')) for v in mask_p]
            line_img = apply_mask(img_c, mask_pts)
            
            if binarize:
                howe_line_img = apply_mask(howe_img, mask_pts)     # Hopefully this works even though Howe binarization takes out a few pixels
                simplebin_line_img = apply_mask(simplebin_img, mask_pts)
            
            imageBin = cv2.imencode('.png', line_img)[1].tostring()
            if binarize:
                howe_imageBin = cv2.imencode('.png', howe_line_img)[1].tostring()
                simplebin_imageBin = cv2.imencode('.png', simplebin_line_img)[1].tostring()
            
            if not checkImageIsValid(imageBin):
                print('%s is not a valid image' % img_file)
                continue
                
            if binarize:
                if not (checkImageIsValid(howe_imageBin) and checkImageIsValid(simplebin_imageBin)):
                    print('%s is not a valid image in howe or sauvola binarization' % image['image_file'])
                    continue
            if not test:
                annotation = gt
                label = annotation.encode('utf-8')
            
            imageKey = 'image-%09d' % cnt
            labelKey = 'label-%09d' % cnt
            fileKey = 'file-%09d' % cnt
            
            if binarize:
                howe_imageKey = 'howe-image-%09d' % cnt
                simplebin_imageKey = 'simplebin-image-%09d' % cnt

            print imageKey
            if binarize:
                print howe_imageKey
                print simplebin_imageKey

            cache[imageKey] = imageBin
            if binarize:
                cache[howe_imageKey] = howe_imageBin
                cache[simplebin_imageKey] = simplebin_imageBin
            if not test:
                cache[labelKey] = label
            cache[fileKey] = os.path.basename(img_file)
            
            if cnt % 1000 == 0:
                writeCache(env, cache)
                cache = {}
                print('Written %d' % (cnt))

            cnt += 1

    nSamples = cnt - 1
    cache['num-samples'] = str(nSamples)
    writeCache(env, cache)
    print('Created dataset with %d samples' % nSamples)

# read into LMDB dataset from XML 
def lmdb_dataset_read(data_dir, output_path, binarize = False, howe_dir = False, simplebin_dir = False, image_dir = False, add_pixel = False):

    env = lmdb.open(output_path, map_size=1099511627776)
    images = page_images(data_dir)
    # print images

    cache = {}
    cnt = 1

    alpha_text = u'' #'0123456789abcdefghijklmnopqrstuvwxyz'
    alphabet = []
    for c in alpha_text:
        alphabet.append(c)


    for image in images:
        print image
        file_image = os.path.join(data_dir,'Images',image.Page.get('imageFilename'))
        print(file_image)
        image['data'] = cv2.imread(file_image)
        page_img = cv2.imread(file_image)
        if binarize:
            howe_img = cv2.imread(os.path.join(howe_dir, os.path.basename(file_image).lower().partition(".jpg")[0] + "_howe.jpg"))
            simplebin_img = cv2.imread(os.path.join(simplebin_dir, os.path.basename(file_image).lower().partition(".jpg")[0] + "_simplebin.jpg"))

        for region in image.Page.TextRegion:
            print 'region'
            print str(region.tag)

            line_tags = [c.tag.split('}')[1] for c in region.getchildren()]

            if any('TextLine' in l for l in line_tags):
                for line in region.TextLine:
                    print 'line '+line.get('id')
                    print str(line.Coords.get('points'))
                    data = line.Coords.get('points')
                    pts = [tuple(int(x) for x in v.split(',')) for v in data.split()]
                    print("Image shape")
                    print(page_img.shape)
                    line_img = apply_mask(page_img, pts, add_pixel)
                    if binarize:
                        howe_line_img = apply_mask(howe_img, pts, add_pixel)     # Hopefully this works even though Howe binarization takes out a few pixels
                        simplebin_line_img = apply_mask(simplebin_img, pts, add_pixel)
                    
                    line_file_name = '_'.join([os.path.basename(file_image).partition('.')[0], line.get('id')])
                    print 'line_file_name: ' + line_file_name
                    if image_dir:
                        cv2.imwrite(os.path.join(image_dir, line_file_name + ".jpg"), line_img)
                    
                    imageBin = cv2.imencode('.png', line_img)[1].tostring()
                    if binarize:
                        howe_imageBin = cv2.imencode('.png', howe_line_img)[1].tostring()
                        simplebin_imageBin = cv2.imencode('.png', simplebin_line_img)[1].tostring()

                    if not checkImageIsValid(imageBin):
                        print('%s is not a valid image' % image['image_file'])
                        continue
                    if binarize:
                        if not (checkImageIsValid(howe_imageBin) and checkImageIsValid(simplebin_imageBin)):
                            print('%s is not a valid image in howe or sauvola binarization' % image['image_file'])
                            continue
                    
                   
                    mini_line_tags = [c.tag.split('}')[1] for c in line.getchildren()]
                    annotation = line.TextEquiv.Unicode.text if any('TextEquiv' in l for l in mini_line_tags) else u''
                    
                    if annotation is None:
                        annotation = u''
                    print("Printing apparent unicode!")
                    print(annotation)

                    label = annotation.encode('utf-8')
                    
                    print("Printing encoded unicode!")
                    print(label)

                    for c in annotation:
                        if not c in alphabet:
                            alphabet.append(c)

                    imageKey = 'image-%09d' % cnt
                    fileKey = 'file-%09d' % cnt
                    if binarize:
                        howe_imageKey = 'howe-image-%09d' % cnt
                        simplebin_imageKey = 'simplebin-image-%09d' % cnt
                    labelKey = 'label-%09d' % cnt

                    print imageKey
                    if binarize:
                        print howe_imageKey
                        print simplebin_imageKey

                    cache[imageKey] = imageBin
                    cache[fileKey] = line_file_name
                    if binarize:
                        cache[howe_imageKey] = howe_imageBin
                        cache[simplebin_imageKey] = simplebin_imageBin
                    cache[labelKey] = label
                    
                    if cnt % 1000 == 0:
                        writeCache(env, cache)
                        cache = {}
                        print('Written %d' % (cnt))

                    line['database_id'] = cnt

                    cnt += 1

    nSamples = cnt - 1
    cache['num-samples'] = str(nSamples)
    writeCache(env, cache)
    print('Created dataset with %d samples' % nSamples)

    alpha_text = u''.join(alphabet)
    with io.open("alphabet.txt", "w", encoding=encoding) as text_file:
        text_file.write(alpha_text)




def extract_strips(data_dir, output_path):  # example of cutting pieces of images out (unused)

    # env = lmdb.open(output_path, map_size=1099511627776)
    images = page_images(data_dir)
    print images

    cache = {}
    cnt = 1


    for image in images:
        print image
        file_image = os.path.join(data_dir,'Images',image.Page.get('imageFilename'))
        image['data'] = cv2.imread(file_image)
        page_img = cv2.imread(file_image)
        # page_img = image['data']

        for region in image.Page.TextRegion:
            print 'region'
            print str(region.tag)

            line_tags = [c.tag.split('}')[1] for c in region.getchildren()]

            if any('TextLine' in l for l in line_tags):
                for line in region.TextLine:
                    print 'line '+line.get('id')
                    print str(line.Coords.get('points'))
                    data = line.Coords.get('points')
                    pts = [tuple(int(x) for x in v.split(',')) for v in data.split()]
                    pts = np.array(pts, np.int32)
                    xmin = min(pts, key=lambda x: x[0])[0]
                    xmax = max(pts, key=lambda x: x[0])[0]

                    ymin = min(pts, key=lambda x: x[1])[1]
                    ymax = max(pts, key=lambda x: x[1])[1]

                    updated_pts = [(p[0] - xmin, p[1] - ymin) for p in pts]
                    line_img = page_img[ymin:ymax, xmin:xmax].copy()
                    # http://stackoverflow.com/a/15343106/3479446
                    mask = np.zeros(line_img.shape, dtype=np.uint8)
                    roi_corners = np.array([updated_pts], dtype=np.int32)

                    channel_count = 1
                    if len(line_img.shape) > 2:
                        channel_count = line_img.shape[2]

                    ignore_mask_color = (255,) * channel_count
                    cv2.fillPoly(mask, roi_corners, ignore_mask_color)
                    line_img[mask == 0] = 255
                    line['data'] = line_img

                    imageKey = 'image-%09d' % cnt
                    cv2.imwrite(os.path.join(output_path, imageKey + '.png'), line_img)
                    cnt += 1



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True, help='path to dataset')
    parser.add_argument('--output_dir', required=True, help='path to lmdb database output')
    parser.add_argument('--output_image_dir', type=str, default="None", help='path to cropped image output if desired')
    parser.add_argument('--xml', action='store_true', help='whether the data are organized in /Images and /Pages subdirectories with PAGE segmentation file format')
    parser.add_argument('--icfhr', action='store_true', help='whether the data are organized according to 2018 ICFHR Handwriting Recognition Competition format')
    parser.add_argument('--russell', action='store_true', help='whether the data are organized according whole page russell journal')
    parser.add_argument('--files_include', help='File of filenames to selectively include in the lmdb database from data_dir')
    parser.add_argument('--binarize', action='store_true', help='whether to include binarized data in lmdb database')
    parser.add_argument('--howe_dir', help='path to howe binarized dataset')
    parser.add_argument('--simplebin_dir', help='path to sauvola binarized dataset')
    parser.add_argument('--test', action='store_true', help='whether to data is a test dataset (includes no ground truth text)')
    parser.add_argument('--add_pixel', action='store_true', help='whether to include extra pixels along y-axis in line segmentation')
    parser.add_argument('--n_pixels', type=int, default=0, help='How many extra pixels to include')
    opt = parser.parse_args()
    print("Running with options:", opt)

    if not os.path.isdir(opt.output_dir):
        os.system('mkdir -p {0}'.format(opt.output_dir))
    if not (opt.output_image_dir == "None") and not os.path.isdir(opt.output_image_dir):
        os.system('mkdir -p {0}'.format(opt.output_image_dir))

    if opt.xml:
        lmdb_dataset_read(opt.data_dir, opt.output_dir, binarize = opt.binarize, howe_dir = opt.howe_dir, simplebin_dir = opt.simplebin_dir, image_dir = opt.output_image_dir if not opt.output_image_dir == "None" else False, add_pixel = opt.n_pixels if opt.add_pixel else False)
    elif opt.icfhr:
        if opt.files_include:
            with open(opt.files_include, "r") as include_file:
                icfhr_dataset_read(opt.data_dir, opt.output_dir, include_file.read().split(), binarize = opt.binarize, howe_dir = opt.howe_dir, simplebin_dir = opt.simplebin_dir, test=opt.test)
        else:
            icfhr_dataset_read(opt.data_dir, opt.output_dir, binarize = opt.binarize, howe_dir = opt.howe_dir, simplebin_dir = opt.simplebin_dir, test=opt.test)
    elif opt.russell:
        russell_page_journal(opt.data_dir, opt.output_dir)
    else:
        simple_dataset_from_dir(opt.data_dir, opt.output_dir) 