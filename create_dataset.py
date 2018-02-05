import os
import lmdb # install lmdb by "pip install lmdb"
import cv2
import numpy as np
from tool.parse_xml import import_images
import sys

def checkImageIsValid(imageBin):
    if imageBin is None:
        return False
    imageBuf = np.fromstring(imageBin, dtype=np.uint8)
    img = cv2.imdecode(imageBuf, cv2.IMREAD_GRAYSCALE)
    imgH, imgW = img.shape[0], img.shape[1]
    if imgH * imgW == 0:
        return False
    return True


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

# def create_read_dataset(outputPath): #, imagePathList, labelList, lexiconList=None, checkValid=True
#     """
#     Create LMDB dataset for CRNN training.
#     ARGS:
#         outputPath    : LMDB output path
#         imagePathList : list of image path
#         labelList     : list of corresponding groundtruth texts
#         lexiconList   : (optional) list of lexicon lists
#         checkValid    : if true, check the validity of every image
#     """
#     # assert(len(imagePathList) == len(labelList))
#     # nSamples = len(imagePathList)
#     env = lmdb.open(outputPath, map_size=1099511627776)
#     cache = {}
#     cnt = 1
#     for i in xrange(nSamples):
#         # imagePath = imagePathList[i]
#         # print imagePath
#         # label = labelList[i]
#         # if not os.path.exists(imagePath):
#         #     print('%s does not exist' % imagePath)
#         #     continue
#         # with open(imagePath, 'r') as f:
#         #     imageBin = f.read()
#         # if checkValid:
#         #     if not checkImageIsValid(imageBin):
#         #         print('%s is not a valid image' % imagePath)
#         #         continue
#
#         imageKey = 'image-%09d' % cnt
#         labelKey = 'label-%09d' % cnt
#         cache[imageKey] = imageBin
#         cache[labelKey] = label
#         if lexiconList:
#             lexiconKey = 'lexicon-%09d' % cnt
#             cache[lexiconKey] = ' '.join(lexiconList[i])
#         if cnt % 1000 == 0:
#             writeCache(env, cache)
#             cache = {}
#             print('Written %d / %d' % (cnt, nSamples))
#         cnt += 1
#     nSamples = cnt-1
#     cache['num-samples'] = str(nSamples)
#     writeCache(env, cache)
#     print('Created dataset with %d samples' % nSamples)

def crnn_dataset():
    imagePathList = []
    labelList = []
    image_dir = 'data/dataset/images/train'
    files = os.listdir(image_dir)
    for file in files:
        image_path = file
        imagePathList.append(os.path.join(image_dir,image_path))
        label = file.split('_')[1]
        labelList.append(label)

    # createDataset("data/dataset/lmdb/train",imagePathList, labelList)
    createDataset("data/dataset/lmdb/train", imagePathList, labelList)


def read_dataset(data_dir, output_path):

    env = lmdb.open(output_path, map_size=1099511627776)
    images = import_images(data_dir)
    print images

    cache = {}
    cnt = 1

    alpha_text = '0123456789abcdefghijklmnopqrstuvwxyz'
    alphabet = []
    for c in alpha_text:
        alphabet.append(c)


    for image in images:
        print image

        image['data'] = cv2.imread(image['image_file'])
        page_img = image['data']

        for line in image['lines']:
            print image['image_file']

            pts = line['bounding_poly']
            pts = np.array(pts, np.int32)

            xmin = min(pts, key=lambda x: x[0])[0]
            xmax = max(pts, key=lambda x: x[0])[0]

            ymin = min(pts, key=lambda x: x[1])[1]
            ymax = max(pts, key=lambda x: x[1])[1]

            updated_pts = [(p[0]-xmin, p[1]-ymin) for p in pts]
            line_img = page_img[ymin:ymax, xmin:xmax].copy()

            #http://stackoverflow.com/a/15343106/3479446
            mask = np.zeros(line_img.shape, dtype=np.uint8)
            roi_corners = np.array([updated_pts], dtype=np.int32)

            channel_count = 1
            if len(line_img.shape) > 2:
                channel_count = line_img.shape[2]

            ignore_mask_color = (255,)*channel_count
            cv2.fillPoly(mask, roi_corners, ignore_mask_color)
            line_img[mask == 0] = 255
            line['data'] = line_img
            imageBin = cv2.imencode('.png', line['data'])[1].tostring()

            if not checkImageIsValid(imageBin):
                print('%s is not a valid image' % image['image_file'])
                continue

            label = line['ground_truth'].encode('utf-8')

            for c in label:
                if not c in alphabet:
                    alphabet.append(c)


            # if not label.isalnum():
            #     print label
            #     sys.exit(0)

            imageKey = 'image-%09d' % cnt
            labelKey = 'label-%09d' % cnt

            print imageKey

            cache[imageKey] = imageBin
            cache[labelKey] = label
            if cnt % 1000 == 0:
                writeCache(env, cache)
                cache = {}
                print('Written %d' % (cnt))

            line['database_id'] = cnt

            cnt += 1

    nSamples = cnt-1
    cache['num-samples'] = str(nSamples)
    writeCache(env, cache)
    print('Created dataset with %d samples' % nSamples)

    alpha_text = ''.join(alphabet)

    with open("alphabet.txt", "w") as text_file:
        text_file.write(alpha_text)


if __name__ == '__main__':
    # data_dir = '/Users/oliver/projects/datasets/htr-small'
    # output_path = 'data/lmdb/val'


    data_dir = sys.argv[1]
    output_path = sys.argv[2]

    read_dataset(data_dir,output_path)