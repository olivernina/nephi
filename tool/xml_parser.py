from lxml import objectify
import os
import io
#encoding="utf-8"

#from lxml import etree
#from lxml.etree import fromstring

#if request.POST:
#    xml = request.POST['xml'].encode('utf-8')
#    parser = etree.XMLParser(ns_clean=True, recover=True, encoding='utf-8')
#    h = fromstring(xml, parser=parser)


# I should follow these instructions to do unicode correctly with the READ XML data.
#http://lxml.de/FAQ.html#why-can-t-lxml-parse-my-xml-from-unicode-strings
    
#Can lxml parse from file objects opened in unicode/text mode?
#Technically, yes. However, you likely do not want to do that, because it is extremely inefficient. The text encoding that libxml2 uses internally is UTF-8, so parsing from a Unicode file means that Python first reads a chunk of data from the file, then decodes it into a new buffer, and then copies it into a new unicode string object, just to let libxml2 make yet another copy while encoding it down into UTF-8 in order to parse it. It's clear that this involves a lot more recoding and copying than when parsing straight from the bytes that the file contains.

#If you really know the encoding better than the parser (e.g. when parsing HTML that lacks a content declaration), then instead of passing an encoding parameter into the file object when opening it, create a new instance of an XMLParser or HTMLParser and pass the encoding into its constructor. Afterwards, use that parser for parsing, e.g. by passing it into the etree.parse(file, parser) function. Remember to open the file in binary mode (mode="rb"), or, if possible, prefer passing the file path directly into parse() instead of an opened Python file object.



def page_images(data_folder):

    images = []

    images_dir =os.path.join(data_folder,'Images')
    xml_dir = os.path.join(data_folder,'page')

    xml_files = [os.path.join(xml_dir,f) for f in os.listdir(xml_dir) if ".xml" in f.lower()]
    images_files = [os.path.join(images_dir,f) for f in os.listdir(images_dir) if ".jpg" in f.lower()]

    for xml in xml_files:
        print xml
        #with io.open(xml, "r", encoding=encoding) as file:
        file  = open(xml)
        xml_string= file.read()
        #xml_string = xml_string_uni.encode(encoding)
        page = objectify.fromstring(xml_string) # convert from xml string to "python type object"

        images.append(page)

    return images

if __name__=='__main__':
    # e = '<foo><bar><type foobar = "1" /><type foobar = "2" /></bar></foo> '
    # result = xmltodict.parse(e)
    # print result
    # result['bar']

    # read in data from the READ dataset
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
