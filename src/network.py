import torch
import torch.nn as nn
from torch.nn import init
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
from torch.nn.modules.batchnorm import _BatchNorm
import torch.nn.functional as F
# from functools import partial
# from src.batchnorm import SynchronizedBatchNorm2d
# BN_MOM = 3e-4
# norm_layer = partial(SynchronizedBatchNorm2d, momentum=BN_MOM)


class BaseNetwork(nn.Module):
    def __init__(self):
        super(BaseNetwork, self).__init__()

    def init_weights(net, init_type='normal', init_gain=0.02):
        """Initialize network weights.
        init_type: normal | xavier | kaiming | orthogonal
        Parameters:
            net (network)   -- network to be initialized
            init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
            init_gain (float)    -- scaling factor for normal, xavier and orthogonal.
        https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py #Line67
        """
        def init_func(m):  # define the initialization function
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    init.normal_(m.weight.data, 0.0, init_gain)
                elif init_type == 'xavier':
                    init.xavier_normal_(m.weight.data, gain=init_gain)
                elif init_type == 'kaiming':
                    init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    init.orthogonal_(m.weight.data, gain=init_gain)
                else:
                    raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)
            elif classname.find('BatchNorm2d') != -1:
                # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
                init.normal_(m.weight.data, 1.0, init_gain)
                init.constant_(m.bias.data, 0.0)

        print('initialize network with %s' % init_type)
        net.apply(init_func)  # apply the initialization function <init_func>


class PISNet(BaseNetwork):
    def __init__(self, residual_blocks=8, init_weights=True):
        super(PISNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True)
        )

        blocks = []
        for _ in range(residual_blocks):
            block = ResnetBlock(dim=256, dilation=1)
            blocks.append(block)
        self.middle = nn.Sequential(*blocks)

        self.tfrm = EMAU(256, 128, stage_num=16)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, padding=1),
        )

        if init_weights:
            self.init_weights()

    def forward(self, x):

        x = self.encoder(x)
        x = self.middle(x)
        x, mu = self.tfrm(x)
        x = self.decoder(x)
        x = torch.sigmoid(x)

        return x, mu


class ResnetBlock(nn.Module):
    def __init__(self, dim, dilation=1):
        super(ResnetBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, stride=1, padding=1, dilation=dilation),
            nn.ReLU(True),
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, stride=1, padding=1, dilation=1),
        )

    def forward(self, x):
        out = x + self.conv_block(x)
        # Remove ReLU at the end of the residual block
        # http://torch.ch/blog/2016/02/04/resnets.html
        return out


class EMAU(nn.Module):
    '''The Expectation-Maximization Attention Unit (EMAU).

    https://github.com/XiaLiPKU/EMANet/blob/master/network.py

    Arguments:
        c (int): The input and output channel number.
        k (int): The number of the bases.
        stage_num (int): The iteration number for EM.
    '''

    def __init__(self, c, k, stage_num=3):
        super(EMAU, self).__init__()
        self.stage_num = stage_num

        mu = torch.Tensor(1, c, k)
        mu.normal_(0, math.sqrt(2. / k))  # Init with Kaiming Norm.
        mu = self._l2norm(mu, dim=1)
        self.register_buffer('mu', mu)    # Add persistent buffers to modules

        self.conv1 = nn.Conv2d(c, c, 1)
        # self.conv2 = nn.Sequential(nn.Conv2d(c, c, 1, bias=False),
        #                            norm_layer(c))
        self.conv2 = nn.Conv2d(c, c, 1, bias=False)
        #
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, _BatchNorm):
                m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        idn = x

        # The first 1x1 conv
        x = self.conv1(x)

        # The EM Attention
        b, c, h, w = x.size()
        # print("x", x.size())
        x = x.view(b, c, h * w)        # b * c * n
        mu = self.mu.repeat(b, 1, 1)   # b * c * k
        with torch.no_grad():
            # Start the iteration
            for i in range(self.stage_num):
                # A_E step
                x_t = x.permute(0, 2, 1)  # b * n * c
                # print("xt", x_t.size())
                z = torch.bmm(x_t, mu)    # b * n * k
                # print("z", z.size())
                z = F.softmax(z, dim=2)   # b * n * k
                z_ = z / (1e-6 + z.sum(dim=1, keepdim=True))
                # print("z_", z_.size())
                # A_M step
                mu = torch.bmm(x, z_)     # b * c * k
                mu = self._l2norm(mu, dim=1)
                # print("mu", mu.size())

        # The moving averaging operation is writtern in train.py.

        # A_R step
        z_t = z.permute(0, 2, 1)   # b * k * n
        x = mu.matmul(z_t)         # b * c * n
        x = x.view(b, c, h, w)     # b * c * h * w
        x = F.relu(x, inplace=True)

        # The second 1x1 conv
        x = self.conv2(x)
        x = x + idn
        x = F.relu(x, inplace=True)

        return x, mu

    def _l2norm(self, inp, dim):
        '''Normlize the inp tensor with l2-norm.

        Returns a tensor where each sub-tensor of input along the given dim is
        normalized such that the 2-norm of the sub-tensor is equal to 1.

        Arguments:
            inp (tensor): The input tensor.
            dim (int): The dimension to slice over to get the ssub-tensors.

        Returns:
            (tensor) The normalized tensor.
        '''
        return inp / (1e-6 + inp.norm(dim=dim, keepdim=True))


class IMN(BaseNetwork):
    def __init__(self, dim=1, init_weights=True):
        super(IMN, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True)
        )
        self.blocks1 = nn.Sequential(
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1, eps=1e-05, momentum=0.1),
            nn.ReLU(True),
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1, eps=1e-05, momentum=0.1)
        )
        self.blocks2 = nn.Sequential(
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1, eps=1e-05, momentum=0.1),
            nn.ReLU(True),
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1, eps=1e-05, momentum=0.1)
        )
        self.blocks3 = nn.Sequential(
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1, eps=1e-05, momentum=0.1),
            nn.ReLU(True),
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1, eps=1e-05, momentum=0.1)
        )
        self.blocks4 = nn.Sequential(
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1, eps=1e-05, momentum=0.1),
            nn.ReLU(True),
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1, eps=1e-05, momentum=0.1)
        )
        self.blocks5 = nn.Sequential(
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1, eps=1e-05, momentum=0.1),
            nn.ReLU(True),
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1, eps=1e-05, momentum=0.1)
        )
        self.blocks6 = nn.Sequential(
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1, eps=1e-05, momentum=0.1),
            nn.ReLU(True),
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1, eps=1e-05, momentum=0.1)
        )
        self.blocks7 = nn.Sequential(
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1, eps=1e-05, momentum=0.1),
            nn.ReLU(True),
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1, eps=1e-05, momentum=0.1)
        )
        self.blocks8 = nn.Sequential(
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1, eps=1e-05, momentum=0.1),
            nn.ReLU(True),
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1, eps=1e-05, momentum=0.1)
        )
        self.blocks9 = nn.Sequential(
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1, eps=1e-05, momentum=0.1),
            nn.ReLU(True),
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1, eps=1e-05, momentum=0.1)
        )
        self.blocks10 = nn.Sequential(
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1, eps=1e-05, momentum=0.1),
            nn.ReLU(True),
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1, eps=1e-05, momentum=0.1)
        )
        self.blocks11 = nn.Sequential(
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1, eps=1e-05, momentum=0.1),
            nn.ReLU(True),
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1, eps=1e-05, momentum=0.1)
        )
        self.blocks12 = nn.Sequential(
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1, eps=1e-05, momentum=0.1),
            nn.ReLU(True),
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1, eps=1e-05, momentum=0.1)
        )
        self.blocks13 = nn.Sequential(
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1, eps=1e-05, momentum=0.1),
            nn.ReLU(True),
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1, eps=1e-05, momentum=0.1)
        )
        self.blocks14 = nn.Sequential(
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1, eps=1e-05, momentum=0.1),
            nn.ReLU(True),
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1, eps=1e-05, momentum=0.1)
        )
        self.blocks15 = nn.Sequential(
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1, eps=1e-05, momentum=0.1),
            nn.ReLU(True),
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1, eps=1e-05, momentum=0.1)
        )
        self.blocks16 = nn.Sequential(
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1, eps=1e-05, momentum=0.1),
            nn.ReLU(True),
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1, eps=1e-05, momentum=0.1)
        )

        self.layer18 = nn.Sequential(
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1, eps=1e-05, momentum=0.1)
        )
        self.layer19 = nn.Sequential(
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, stride=1, padding=1),
        )
        if init_weights:
            self.init_weights()

    def forward(self, x):
        x1 = self.layer1(x)

        x2 = x1 + self.blocks1(x1)
        x3 = x2 + self.blocks2(x2)
        x4 = x3 + self.blocks3(x3)
        x5 = x4 + self.blocks4(x4)

        x6 = x5 + self.blocks5(x5)
        x7 = x6 + self.blocks6(x6)
        x8 = x7 + self.blocks7(x7)
        x9 = x8 + self.blocks8(x8)

        x10 = x9 + self.blocks9(x9)
        x11 = x10 + self.blocks10(x10)
        x12 = x11 + self.blocks11(x11)
        x13 = x12 + self.blocks12(x12)

        x14 = x13 + self.blocks13(x13)
        x15 = x14 + self.blocks14(x14)
        x16 = x15 + self.blocks15(x15)
        x17 = x16 + self.blocks16(x16)

        x18 = x1 + self.layer18(x17)
        x = self.layer19(x18)

        return x

