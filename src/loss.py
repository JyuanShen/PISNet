import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


class LossCriterion(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_mse = nn.MSELoss()

    def forward(self, x, y, x0):
        # The Mean Square Error Loss
        loss_mse = self.loss_mse(x, y)

        # The low-rank property is represented by Nuclear-norm, which can be calculated by Singular Values
        loss_lowrank = torch.norm(x[0].squeeze(0), p="nuc", dim=None) / 256

        # The sparse property is represented by the L1-Norm
        loss_sparse = torch.norm(x0-x, p=1, dim=None) / (256 * 256)

        # The total loss
        lamda_1 = 1000
        lamda_2 = 0.1
        lamda_3 = 0.01
        loss_all = lamda_1*loss_mse + lamda_2*loss_lowrank + lamda_3*loss_sparse
        return loss_all, loss_mse, loss_lowrank, loss_sparse


class IMNLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_mse = nn.MSELoss()

    def forward(self, x, y):
        # The Mean Square Error Loss
        loss_mse = self.loss_mse(x, y)
        return loss_mse


# loss_func = LossCriterion()
# a = torch.tensor([[[[1., 2.], [3., 4.]]], [[[1., 2.], [3., 4.]]]])
# b = torch.tensor([[[[0., 0.], [0., 0.]]], [[[0., 0.], [0., 0.]]]])
# # print(a)
# # print(a.shape)
# # print(b)
# # a = a.unsqueeze(0).unsqueeze(1)
# # b = b.unsqueeze(0).unsqueeze(1)
# print(a)
# print(b)
# print(a.shape)
# print(b.shape)
# loss, mse, low, spr = loss_func(a, b)
# print(loss, mse, low, spr)

