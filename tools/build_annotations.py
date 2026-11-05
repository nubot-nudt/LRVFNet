import os
import sys
import xml.etree.ElementTree as ET
from os import getcwd

# allow running as `python tools/build_annotations.py` from the repository root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.common import get_classes

sets=[('2007', 'train'), ('2007', 'val'), ('2007', 'test')]
classes_path        = 'model_data/classes.txt'

classes, _      = get_classes(classes_path)

def convert_annotation(year, image_id, list_file):
    in_file = open('VOCdevkit/VOC%s/Annotations/%s.xml'%(year, image_id))
    tree=ET.parse(in_file)
    root = tree.getroot()

    for obj in root.iter('object'):
        difficult = 0 
        if obj.find('difficult')!=None:
            difficult = obj.find('difficult').text
            
        cls = obj.find('name').text
        if cls not in classes or int(difficult)==1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text), int(xmlbox.find('xmax').text), int(xmlbox.find('ymax').text))
        list_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))

wd = getcwd()
for year, image_set in sets:
    image_ids = open('VOCdevkit/VOC%s/ImageSets/Main/%s.txt'%(year, image_set)).read().strip().split()
    list_file = open('%s_%s.txt'%(year, image_set), 'w')
    for image_id in image_ids:
        list_file.write('%s/VOCdevkit/VOC%s/JPEGImages/%s.jpg'%(wd, year, image_id))  
        # _1 = LiDAR BEV, _2 = radar BEV (paper Fig.2)
        list_file.write(' %s/VOCdevkit/VOC2007/LRVF/%s_1.jpg'% (wd,image_id))
        list_file.write(' %s/VOCdevkit/VOC2007/LRVF/%s_2.jpg'% (wd,image_id))
        convert_annotation(year, image_id, list_file)
        list_file.write('\n')
    list_file.close()