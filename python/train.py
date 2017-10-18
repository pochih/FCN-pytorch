# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

from fcn import VGGNet, FCN32s, FCN16s, FCN8s, FCNs
from CamVid_loader import CamVidDataset

from matplotlib import pyplot as plt
import numpy as np
import time
import os


root_dir   = "CamVid/"
train_file = os.path.join(root_dir, "train.csv")
val_file   = os.path.join(root_dir, "val.csv")

# create dir for save model & score
model_dir = "models"
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
score_dir = "scores"
if not os.path.exists(score_dir):
    os.makedirs(score_dir)

h, w, c    = 720, 960, 3
n_class    = 32

batch_size = 14
epochs     = 300
lr         = 1e-4
momentum   = 0
w_decay    = 1e-4
configs    = "FCNs_batch{}_epoch{}_RMSprop_lr{}_momentum{}_w_decay{}".format(batch_size, epochs, lr, momentum, w_decay)

use_gpu = torch.cuda.is_available()
num_gpu = list(range(torch.cuda.device_count()))

train_data   = CamVidDataset(csv_file=train_file)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=8)

val_data   = CamVidDataset(csv_file=val_file)
val_loader = DataLoader(val_data, batch_size=1, num_workers=8)

vgg_model = VGGNet(requires_grad=True)
fcn_model = FCNs(pretrained_net=vgg_model, classes=n_class)
if use_gpu:
    ts = time.time()
    vgg_model = vgg_model.cuda()
    fcn_model = fcn_model.cuda()
    fcn_model = nn.DataParallel(fcn_model, device_ids=num_gpu)
    print("Finish cuda loading, time elapsed {}".format(time.time() - ts))

criterion = nn.BCELoss()
optimizer = optim.RMSprop(fcn_model.parameters(), lr=lr, momentum=momentum, weight_decay=w_decay)

scores = np.zeros((epochs, n_class))


def train():
    for epoch in range(epochs):
        ts = time.time()
        for iter, batch in enumerate(train_loader):
            optimizer.zero_grad()

            if use_gpu:
                inputs = Variable(batch['X'].cuda())
                labels = Variable(batch['Y'].cuda())
            else:
                inputs, labels = Variable(batch['X']), Variable(batch['Y'])

            outputs = fcn_model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if iter % 10 == 0:
                print("epoch{}, iter{}, loss: {}".format(epoch, iter, loss.data[0]))
        
        print("Finish epoch {}, time elapsed {}".format(epoch, time.time() - ts))
        torch.save(fcn_model.state_dict(), os.path.join(model_dir, configs))

        val(epoch)


def val(epoch):
    fcn_model.eval()
    total_ious = []
    for iter, batch in enumerate(val_loader):
        if use_gpu:
            inputs = Variable(batch['X'].cuda())
        else:
            inputs = Variable(batch['X'])

        output = fcn_model(inputs)
        output = output.data.cpu().numpy()

        _, _, h, w = output.shape
        pred = output.transpose(0, 2, 3, 1).reshape(-1, n_class).argmax(axis=1).reshape(h, w)
        target = batch['l'].cpu().numpy().reshape(h, w)
        total_ious.append(iou(pred, target))

    # Calculate average IoU
    total_ious = np.array(total_ious).T  # n_class * val_len
    ious = total_ious.mean(axis=1)
    print("epoch{}, meanIoU: {}, IoUs: {}".format(epoch, ious.mean(), ious))
    scores[epoch] = ious
    np.save(os.path.join(score_dir, configs), scores)


# borrow functions from https://github.com/Kaixhin/FCN-semantic-segmentation/blob/master/main.py
# Calculates class intersections over unions
def iou(pred, target):
    ious = []
    for cls in range(n_class):
        pred_inds = pred == cls
        target_inds = target == cls
        intersection = pred_inds[target_inds].sum()
        union = pred_inds.sum() + target_inds.sum() - intersection
        ious.append(float(intersection) / max(union, 1))
        # print("cls", cls, pred_inds.sum(), target_inds.sum(), intersection, float(intersection) / max(union, 1))
    return ious


if __name__ == "__main__":
    val(0)
    train()
