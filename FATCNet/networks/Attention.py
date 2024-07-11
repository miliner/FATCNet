import torch
import torch.nn as nn


class SCSEModule(nn.Module):
    def __init__(self, ch, re=16):
        super().__init__()
        self.cSE = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                 nn.Conv2d(ch,ch//re,1),
                                 nn.ReLU(inplace=True),
                                 nn.Conv2d(ch//re,ch,1),
                                 nn.Sigmoid())
        self.sSE = nn.Sequential(nn.Conv2d(ch,ch,1),
                                 nn.Sigmoid())
    def forward(self, x):
        return x * self.cSE(x) + x * self.sSE(x)


class se_module(nn.Module):
    def __init__(self, channel, reduction=16):
        super(se_module, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, int(channel // reduction), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(int(channel // reduction), channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


#  需要重跑V2
class SpatialAttention(nn.Module):
    def __init__(self, img_size):
        super(SpatialAttention, self).__init__()

        if img_size == 256:
            self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)
        else:
            self.conv1 = nn.Conv2d(2, 1, kernel_size=3, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        _, _, w, h = x.shape
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        y = torch.cat([max_pool, avg_pool], dim=1)
        if w == 128:
            y = self.conv(y)
        else:
            y = self.conv1(y)
        return x * self.sigmoid(y)


class SpatialAttention_small(nn.Module):
    def __init__(self, img_size):
        super(SpatialAttention_small, self).__init__()

        if img_size == 256:
            self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)
        else:
            self.conv1 = nn.Conv2d(2, 1, kernel_size=3, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        _, _, w, h = x.shape
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        if w == 128:
            y = torch.cat([max_pool, avg_pool], dim=1)
            y = self.conv(y)
        else:
            y = torch.cat([avg_pool, max_pool], dim=1)
            y = self.conv1(y)
        return x * self.sigmoid(y)
