# -*- coding: utf-8 -*-

from __future__ import print_function

from collections import namedtuple
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import scipy.misc
import random
import os


#############################
    # global variables #
#############################
root_dir  = "CityScapes/"

label_dir = os.path.join(root_dir, "gtFine")
train_dir = os.path.join(label_dir, "train")
val_dir   = os.path.join(label_dir, "val")
test_dir  = os.path.join(label_dir, "test")

# create dir for label index
label_idx_dir = os.path.join(root_dir, "Labeled_idx")
train_idx_dir = os.path.join(label_idx_dir, "train")
val_idx_dir   = os.path.join(label_idx_dir, "val")
test_idx_dir  = os.path.join(label_idx_dir, "test")
for dir in [train_idx_dir, val_idx_dir, test_idx_dir]:
    if not os.path.exists(dir):
        os.makedirs(dir)

train_file = os.path.join(root_dir, "train.csv")
val_file   = os.path.join(root_dir, "val.csv")
test_file  = os.path.join(root_dir, "test.csv")

color2index = {}

Label = namedtuple('Label', [
                   'name', 
                   'id', 
                   'trainId', 
                   'category', 
                   'categoryId', 
                   'hasInstances', 
                   'ignoreInEval', 
                   'color'])

labels = [
    #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
    Label(  'unlabeled'            ,  0 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'ego vehicle'          ,  1 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'rectification border' ,  2 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'out of roi'           ,  3 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'static'               ,  4 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'dynamic'              ,  5 ,      255 , 'void'            , 0       , False        , True         , (111, 74,  0) ),
    Label(  'ground'               ,  6 ,      255 , 'void'            , 0       , False        , True         , ( 81,  0, 81) ),
    Label(  'road'                 ,  7 ,        1 , 'flat'            , 1       , False        , False        , (128, 64,128) ),
    Label(  'sidewalk'             ,  8 ,        2 , 'flat'            , 1       , False        , False        , (244, 35,232) ),
    Label(  'parking'              ,  9 ,      255 , 'flat'            , 1       , False        , True         , (250,170,160) ),
    Label(  'rail track'           , 10 ,      255 , 'flat'            , 1       , False        , True         , (230,150,140) ),
    Label(  'building'             , 11 ,        3 , 'construction'    , 2       , False        , False        , ( 70, 70, 70) ),
    Label(  'wall'                 , 12 ,        4 , 'construction'    , 2       , False        , False        , (102,102,156) ),
    Label(  'fence'                , 13 ,        5 , 'construction'    , 2       , False        , False        , (190,153,153) ),
    Label(  'guard rail'           , 14 ,      255 , 'construction'    , 2       , False        , True         , (180,165,180) ),
    Label(  'bridge'               , 15 ,      255 , 'construction'    , 2       , False        , True         , (150,100,100) ),
    Label(  'tunnel'               , 16 ,      255 , 'construction'    , 2       , False        , True         , (150,120, 90) ),
    Label(  'pole'                 , 17 ,        6 , 'object'          , 3       , False        , False        , (153,153,153) ),
    Label(  'polegroup'            , 18 ,      255 , 'object'          , 3       , False        , True         , (153,153,153) ),
    Label(  'traffic light'        , 19 ,        7 , 'object'          , 3       , False        , False        , (250,170, 30) ),
    Label(  'traffic sign'         , 20 ,        8 , 'object'          , 3       , False        , False        , (220,220,  0) ),
    Label(  'vegetation'           , 21 ,        9 , 'nature'          , 4       , False        , False        , (107,142, 35) ),
    Label(  'terrain'              , 22 ,       10 , 'nature'          , 4       , False        , False        , (152,251,152) ),
    Label(  'sky'                  , 23 ,       11 , 'sky'             , 5       , False        , False        , ( 70,130,180) ),
    Label(  'person'               , 24 ,       12 , 'human'           , 6       , True         , False        , (220, 20, 60) ),
    Label(  'rider'                , 25 ,       13 , 'human'           , 6       , True         , False        , (255,  0,  0) ),
    Label(  'car'                  , 26 ,       14 , 'vehicle'         , 7       , True         , False        , (  0,  0,142) ),
    Label(  'truck'                , 27 ,       15 , 'vehicle'         , 7       , True         , False        , (  0,  0, 70) ),
    Label(  'bus'                  , 28 ,       16 , 'vehicle'         , 7       , True         , False        , (  0, 60,100) ),
    Label(  'caravan'              , 29 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0, 90) ),
    Label(  'trailer'              , 30 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0,110) ),
    Label(  'train'                , 31 ,       17 , 'vehicle'         , 7       , True         , False        , (  0, 80,100) ),
    Label(  'motorcycle'           , 32 ,       18 , 'vehicle'         , 7       , True         , False        , (  0,  0,230) ),
    Label(  'bicycle'              , 33 ,       19 , 'vehicle'         , 7       , True         , False        , (119, 11, 32) ),
    Label(  'license plate'        , -1 ,       -1 , 'vehicle'         , 7       , False        , True         , (  0,  0,142) ),
]


def parse_label():
    # change label to class index
    color2index[(0,0,0)] = 0  # add an void class 
    for obj in labels:
        if obj.ignoreInEval:
            continue
        idx   = obj.trainId
        label = obj.name
        color = obj.color
        color2index[color] = idx

    # parse train, val, test data    
    for label_dir, index_dir, csv_file in zip([train_dir, val_dir, test_dir], [train_idx_dir, val_idx_dir, test_idx_dir], [train_file, val_file, test_file]):
        f = open(csv_file, "w")
        f.write("img,label\n")
        for city in os.listdir(label_dir):
            city_dir = os.path.join(label_dir, city)
            city_idx_dir = os.path.join(index_dir, city)
            data_dir = city_dir.replace("gtFine", "leftImg8bit")
            if not os.path.exists(city_idx_dir):
                os.makedirs(city_idx_dir)
            for filename in os.listdir(city_dir):
                if 'color' not in filename:
                    continue
                lab_name = os.path.join(city_idx_dir, filename)
                img_name = filename.split("gtFine")[0] + "leftImg8bit.png"
                img_name = os.path.join(data_dir, img_name)
                f.write("{},{}.npy\n".format(img_name, lab_name))

                if os.path.exists(lab_name + '.npy'):
                    print("Skip %s" % (filename))
                    continue
                print("Parse %s" % (filename))
                img = os.path.join(city_dir, filename)
                img = scipy.misc.imread(img, mode='RGB')
                height, weight, _ = img.shape
        
                idx_mat = np.zeros((height, weight))
                for h in range(height):
                    for w in range(weight):
                        color = tuple(img[h, w])
                        try:
                            index = color2index[color]
                            idx_mat[h, w] = index
                        except:
                            # no index, assign to void
                            idx_mat[h, w] = 19
                idx_mat = idx_mat.astype(np.uint8)
                np.save(lab_name, idx_mat)
                print("Finish %s" % (filename))


'''debug function'''
def imshow(img, title=None):
    try:
        img = mpimg.imread(img)
        imgplot = plt.imshow(img)
    except:
        plt.imshow(img, interpolation='nearest')

    if title is not None:
        plt.title(title)
    
    plt.show()


if __name__ == '__main__':
    parse_label()
