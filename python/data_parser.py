# -*- coding: utf-8 -*-

from __future__ import print_function
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import scipy.misc
import os


#############################
    # global variables #
#############################
# load CamVid data
root_dir          = "CamVid/"
label_dir         = os.path.join(root_dir, "LabeledApproved_full")  # train label
label_colors_file = os.path.join(root_dir, "label_colors.txt")      # color to label

# create dir for label index
label_idx_dir = os.path.join(root_dir, "Labeled_idx")
if not os.path.exists(label_idx_dir):
    os.makedirs(label_idx_dir)

label2color = {}
color2label = {}
label2index = {}
index2label = {}


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
    f = open(label_colors_file, "r").read().split("\n")[:-1]  # ignore the last empty line
    for idx, line in enumerate(f):
        label = line.split()[-1]
        color = tuple([int(x) for x in line.split()[:-1]])
        print(label, color)
        label2color[label] = color
        color2label[color] = label
        label2index[label] = idx
        index2label[idx]   = label
        # rgb = np.zeros((255, 255, 3), dtype=np.uint8)
        # rgb[..., 0] = color[0]
        # rgb[..., 1] = color[1]
        # rgb[..., 2] = color[2]
        # imshow(rgb, title=label)
    
    for idx, name in enumerate(os.listdir(label_dir)):
        filename = os.path.join(label_idx_dir, name)
        if os.path.exists(filename + '.npy'):
            print("Skip %s" % (name))
            continue
        print("Parse %s" % (name))
        img = os.path.join(label_dir, name)
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
                    print("error: img:%s, h:%d, w:%d" % (name, h, w))
        np.save(filename, idx_mat)
        print("Finish %s" % (name))
    
    img = os.path.join(label_dir, os.listdir(label_dir)[0])
    img = scipy.misc.imread(img, mode='RGB')
    print('img.shape =', img.shape)
    
    test_cases = [(555, 405), (0, 0), (380, 645), (577, 943)]
    test_ans   = ['Car', 'Building', 'Truck_Bus', 'Car']
    for idx, t in enumerate(test_cases):
        color = img[t]
        assert color2label[tuple(color)] == test_ans[idx]

