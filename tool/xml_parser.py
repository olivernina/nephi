import xmltodict
from lxml import objectify
from collections import defaultdict
import os


def page_images(data_folder):

    images = []

    images_dir =os.path.join(data_folder,'Images')
    xml_dir = os.path.join(data_folder,'page')

    xml_files = [os.path.join(xml_dir,f) for f in os.listdir(xml_dir)]
    images_files = [os.path.join(images_dir,f) for f in os.listdir(images_dir)]

    for xml in xml_files:
        print xml
        file  = open(xml)
        xml_string = file.read()
        page = objectify.fromstring(xml_string)

        images.append(page)

    return images

if __name__=='__main__':
    # e = '<foo><bar><type foobar = "1" /><type foobar = "2" /></bar></foo> '
    # result = xmltodict.parse(e)
    # print result
    # result['bar']

    data_folder = '/Users/oliver/projects/datasets/htr-small'

    images_dir =os.path.join(data_folder,'Images')
    xml_dir = os.path.join(data_folder,'page')

    xml_files = [os.path.join(xml_dir,f) for f in os.listdir(xml_dir)]
    images_files = [os.path.join(images_dir,f) for f in os.listdir(images_dir)]

    for xml in xml_files:
        print xml
        file  = open(xml)
        xml_string = file.read()

        # print xml_string
        root = objectify.fromstring(xml_string)
        # print root
        # print root.Page.TextRegion
        # print objectify.SubElement(root,'Page')
        for region in root.Page.TextRegion:
            print 'region'
            print str(region.tag)

            line_tags = [c.tag.split('}')[1] for c in region.getchildren()]

            if any('TextLine' in l for l in line_tags):
                for line in region.TextLine:
                    print 'line '+line.get('id')
                    print str(line.Coords.get('points'))
            #
            #     line.Coords.get('points')

    # count = defaultdict(int)
    #
    # root = objectify.fromstring(e)
    #
    # for item in root.bar.type:
    #     print item.attrib.get("foobar")

    # print dict(count)