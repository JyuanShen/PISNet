import torch
from torchvision import transforms
from torch.utils.data import Dataset
import os
import numpy as np
import cv2
import scipy.io as scio


class readDataset(Dataset):
    def __init__(self, xdir, ydir, transform1=None, transform2=None):
        self.xdir = xdir
        self.ydir = ydir
        self.xitem = os.listdir(self.xdir)
        self.yitem = os.listdir(self.ydir)
        self.xtransform = transform1
        self.ytransform = transform2

    def __len__(self):
        if len(self.xitem) == len(self.yitem):
            return len(self.xitem)
        else:
            print("Error: The number of elements in two items does not match.")
            return None

    def __getitem__(self, index):
        xname = self.xitem[index]
        xpath = os.path.join(self.xdir, xname)
        x = scio.loadmat(xpath)
        xdata = x.get('x')
        xdata = torch.from_numpy(xdata).type(torch.FloatTensor) / 3500.  # conversion ratio
        xdata = torch.unsqueeze(xdata, dim=0)

        yname = self.yitem[index]
        ypath = os.path.join(self.ydir, yname)
        y = scio.loadmat(ypath)
        ydata = y.get('y')
        ydata = torch.from_numpy(ydata).type(torch.FloatTensor) / 1100.  # conversion ratio
        ydata = torch.unsqueeze(ydata, dim=0)

        # if self.xtransform:
        #     xdata = self.xtransform(xdata)
        # if self.ytransform:
        #     ydata = self.xtransform(ydata)

        return xdata, ydata


class readDataset1(Dataset):
    def __init__(self, dir):
        self.dir = dir
        self.item = os.listdir(self.dir)

    def __len__(self):
        return len(self.item)

    def __getitem__(self, index):
        name = self.item[index]
        # print(name)
        path = os.path.join(self.dir, name)
        data = scio.loadmat(path)
        data = data.get('x')
        data = torch.from_numpy(data).type(torch.FloatTensor)

        return data

