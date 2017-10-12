# -*- coding: utf-8 -*-

from __future__ import print_function
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import scipy.misc
import os


###########################
# global variables #
###########################
data_dir = "CamVid/"
test_data_dir = os.path.join(data_dir, "LabeledApproved_full")
label_colors_file = os.path.join(data_dir, "label_colorsSorted.txt")

test_data_idx_dir = os.path.join(data_dir, "Labeled_idx")
if not os.path.exists(test_data_idx_dir):
    os.makedirs(test_data_idx_dir)

label2color = {}
color2label = {}
label2index = {}
index2label = {}


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
    f = open(label_colors_file, "r").read().split("\n")
    for idx, line in enumerate(f):
        label = line.split()[0]
        color = tuple([int(x) for x in line.split()[1:]])
        label2color[label] = color
        color2label[color] = label
        label2index[label] = idx
        index2label[idx]   = label
        # rgb = np.zeros((255, 255, 3), dtype=np.uint8)
        # rgb[..., 0] = color[0]
        # rgb[..., 1] = color[1]
        # rgb[..., 2] = color[2]
        # imshow(rgb, title=label)
    
    for i in os.listdir(test_data_dir):
        filename = os.path.join(test_data_idx_dir, i)
        if os.path.exists(filename + '.npy'):
            continue
        print("Parse %s" % (i))
        img = os.path.join(test_data_dir, i)
        img = scipy.misc.imread(img, mode='RGB')
        height, weight, _ = img.shape
    
        idx_mat = np.zeros((height, weight))
        for h in range(height):
            for w in range(weight):
                color = tuple(img[h, w])
                try:
                    label = color2label[color]
                    index = label2index[label]
                    idx_mat[h, w] = index
                except:
                    print("error: img:%s, h:%d, w:%d" % (i, h, w))
        np.save(filename, idx_mat)
        print("Finish %s" % (i))
    
    img = os.path.join(test_data_dir, os.listdir(test_data_dir)[0])
    img = scipy.misc.imread(img, mode='RGB')
    print(img.shape)
    
    test_cases = [(555, 405), (0, 0), (380, 645), (577, 943)]
    for t in test_cases:
        index = img[t]
        print(index, color2label[tuple(index)])
    
    imshow(img)

