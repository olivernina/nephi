import sys
import os
from os import path, listdir
from os.path import join, isfile
from collections import defaultdict
# import data_utils
import xml.etree.ElementTree
import re
import json

def extract_points(data_string):
    return [tuple(int(x) for x in v.split(',')) for v in data_string.split()]

# http://stackoverflow.com/a/12946675/3479446
def get_namespace(element):
    m = re.match('\{.*\}', element.tag)
    return m.group(0) if m else ''

def readXMLFile(xml_file):
    root = xml.etree.ElementTree.parse(xml_file).getroot()
    namespace = get_namespace(root)

    pages = []
    for page in root.findall(namespace+'Page'):
        pages.append(process_page(page, namespace))

    return pages

def process_page(page, namespace):

    page_out = {}
    regions = []
    lines = []
    for region in page.findall(namespace+'TextRegion'):
        region_out, region_lines = process_region(region, namespace)

        regions.append(region_out)
        lines += region_lines

    page_out['regions'] = regions
    page_out['lines'] = lines

    return page_out

def process_region(region, namespace):

    region_out = {}

    coords = region.find(namespace+'Coords')
    region_out['bounding_poly'] = extract_points(coords.attrib['points'])
    region_out['id'] = region.attrib['id']

    lines = []
    for line in region.findall(namespace+'TextLine'):
        line_out = process_line(line, namespace)
        line_out['region_id'] = region.attrib['id']
        lines.append(line_out)

    return region_out, lines

def process_line(line, namespace):
    errors = []




    line_out = {}

    if 'custom' in line.attrib:
        custom = line.attrib['custom']
        custom = custom.split(" ")
        if "readingOrder" in custom:
            roIdx = custom.index("readingOrder")
            ro = int("".join([v for v in custom[roIdx+1] if v.isdigit()]))
            line_out['read_order'] = ro

    baseline = line.find(namespace+'Baseline')

    if baseline is not None:
        line_out['baseline'] = extract_points(baseline.attrib['points'])
    else:
        errors.append('No baseline')

    coords = line.find(namespace+'Coords')
    line_out['bounding_poly'] = extract_points(coords.attrib['points'])

    ground_truth = line.find(namespace+'TextEquiv').find(namespace+'Unicode').text

    if ground_truth == None or len(ground_truth) == 0:
        errors.append("No ground truth")
        ground_truth = ""

    line_out['ground_truth'] = ground_truth
    if len(errors) > 0:
        line_out['errors'] = errors

    return line_out

def import_images(data_folder):
    image_folder = join(data_folder, 'Images')
    xml_folder = join(data_folder, 'page')

    image_files = [f for f in listdir(image_folder) if isfile(join(image_folder, f))]
    xml_files = [f for f in listdir(xml_folder) if isfile(join(xml_folder, f))]

    # d = defaultdict(dict)
    # for image_file in image_files:
    #     filename = image_file.split('.')[0]
    #     d[filename]['image_file'] = image_file
    #
    # for xml_file in xml_files:
    #     d[data_utils.remove_extension(xml_file)]['xml_file'] = xml_file

    images = []
    for file in image_files:
        # if 'image_file' not in value or 'xml_file' not in value:
        #     continue

        image = {}
        image['image_file'] = join(image_folder, file)

        xml_path = join(xml_folder, file.split('.')[0] + '.xml')
        xmlFileResult = readXMLFile(xml_path)

        assert len(xmlFileResult) == 1

        image['lines'] = xmlFileResult[0]['lines']
        image['regions'] = xmlFileResult[0]['regions']

        images.append(image)

    # return {"images":images}
    return images

import cv2
import numpy as np
if __name__ == '__main__':
    data_folder = 'data/htr-small/'#sys.argv[1]
    # save_file = sys.argv[2]

    images = import_images(data_folder)
    # print images

    for image in images:
        print image

        image['data'] = cv2.imread(image['image_file'])
        img = image['data']

        for line in image['lines']:
            pts = line['bounding_poly']
            pts = np.array(pts, np.int32)

            xmin = min(pts, key=lambda x: x[0])[0]
            xmax = max(pts, key=lambda x: x[0])[0]

            ymin = min(pts, key=lambda x: x[1])[1]
            ymax = max(pts, key=lambda x: x[1])[1]

            updated_pts = [(p[0]-xmin, p[1]-ymin) for p in pts]
            img = img[ymin:ymax, xmin:xmax].copy()

            #http://stackoverflow.com/a/15343106/3479446
            mask = np.zeros(img.shape, dtype=np.uint8)
            roi_corners = np.array([updated_pts], dtype=np.int32)

            channel_count = 1
            if len(img.shape) > 2:
                channel_count = img.shape[2]

            ignore_mask_color = (255,)*channel_count
            cv2.fillPoly(mask, roi_corners, ignore_mask_color)
            img[mask == 0] = 255


            # with open(save_file, 'w') as f:
    #     json.dump(images, f)