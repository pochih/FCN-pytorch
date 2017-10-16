# -*- coding: utf-8 -*-

from __future__ import print_function

from matplotlib import pyplot as plt
import numpy as np
import scipy.misc
import os

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import utils


root_dir          = "CamVid/"
data_dir          = os.path.join(root_dir, "701_StillsRaw_full")  # train data
label_idx_dir     = os.path.join(root_dir, "Labeled_idx")         # train label (transform to class index)

means = (.5, .6, .7)
stds  = (2., 3., 4.)


class CamVidDataset(Dataset):

    def __init__(self, data_dir, label_idx_dir, n_class=32):
        self.data      = [os.path.join(data_dir, d) for d in os.listdir(data_dir)]  # list of filename with type .png
        self.label     = [os.path.join(label_idx_dir, l) for l in os.listdir(label_idx_dir)]  # list of filename with type .npy
        self.means     = means  # mean of three channels after divide to 255
        self.stds      = stds   # std of three channels after divide to 255
        self.n_class   = n_class

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name   = self.data[idx]
        img        = scipy.misc.imread(img_name, mode='RGB')
        label_name = self.label[idx]
        label      = np.load(label_name)

        # convert to tensors
        h, w, _ = img.shape
        img = torch.from_numpy(img).permute(2, 0, 1).float().div(255)
        img[0].sub_(self.means[0]).div_(self.stds[0])
        img[1].sub_(self.means[1]).div_(self.stds[1])
        img[2].sub_(self.means[2]).div_(self.stds[2])
        label = torch.from_numpy(label).long()

        # create one-hot encoding
        target = torch.zeros(self.n_class, h, w)
        for c in range(self.n_class):
            target[c][label == c] = 1

        sample = {'X': img, 'Y': target, 'l': label}

        return sample


def show_batch(batch):
    img_batch = batch['X']
    img_batch[:,0,...].mul_(stds[0]).add_(means[0])
    img_batch[:,1,...].mul_(stds[1]).add_(means[1])
    img_batch[:,2,...].mul_(stds[2]).add_(means[2])
    batch_size = len(img_batch)

    grid = utils.make_grid(img_batch)
    plt.imshow(grid.numpy().transpose((1, 2, 0)))

    plt.title('Batch from dataloader')


if __name__ == "__main__":
    train_data = CamVidDataset(data_dir, label_idx_dir)

    # show a batch
    batch_size = 4
    for i in range(batch_size):
        sample = train_data[i]
        print(i, sample['X'].size(), sample['Y'].size())  

    dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4)

    for i, batch in enumerate(dataloader):
        print(i, batch['X'].size(), batch['Y'].size())
    
        # observe 4th batch
        if i == 3:
            plt.figure()
            show_batch(batch)
            plt.axis('off')
            plt.ioff()
            plt.show()
            break
