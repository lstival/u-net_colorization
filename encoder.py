# Imports

import torch
from torch import nn
import torch.nn.functional
from torch.nn import Dropout, Softmax, Linear, LayerNorm, Conv2d
import math
import copy
import os

# Cuda Devices avalibles
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# CNN Encoder

# BatchNorm is not used because it's not good for ViT
# GroupNorm and LayerNorm is used instead

# Conv with GroupNorm
class CNNencoder_gn(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(16, out_c, eps=1e-6),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        out = self.model(x)
        return out


# Conv with LayerNorm
class CNNencoder_ln(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(23, out_c, eps=1e-6),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        out = self.model(x)
        return out


# CNN Concat with GroupNorm
class Concat_gn(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(16, out_c, eps=1e-6),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x, skip):

        x = torch.cat((x, skip), 1)
        out = self.model(x)
        return out

# CNN concat with LayerNorm
class Concat_ln(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(23, out_c, eps=1e-6),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x, skip):

        x = torch.cat((x, skip), 1)
        out = self.model(x)
        return out

class Enconder(nn.Module):
    def __init__(self, img_size=(512, 768)):
        super().__init__()

        self.pooling = nn.MaxPool2d(kernel_size=2)
        self.upsample = nn.Upsample(scale_factor=2)

        self.conv1_1 = CNNencoder_gn(3, 16)
        self.conv1_2 = CNNencoder_gn(16, 16)
        self.conv2_1 = CNNencoder_gn(16, 32)
        self.conv2_2 = CNNencoder_gn(32, 32)
        self.conv3_1 = CNNencoder_gn(32, 64)
        self.conv3_2 = CNNencoder_gn(64, 64)
        self.conv4_1 = CNNencoder_gn(64, 128)
        self.conv4_2 = CNNencoder_gn(128, 128)
        self.conv5_1 = CNNencoder_gn(128, 256)
        self.conv5_2 = CNNencoder_gn(256, 256)

    def forward(self, x):
        # (B, in_channel, 512, 768)
        c1 = self.conv1_1(x)
        c1 = self.conv1_2(c1)
        # (B, 16, 512, 768)
        p1 = self.pooling(c1)
        # (B, 16, 256, 384)
        c2 = self.conv2_1(p1)
        c2 = self.conv2_2(c2)
        # (B, 16, 256, 384)
        p2 = self.pooling(c2)
        # (B, 32, 128, 192)
        c3 = self.conv3_1(p2)
        c3 = self.conv3_2(c3)
        # (B, 32, 128, 192)
        p3 = self.pooling(c3)
        # (B, 64, 64, 96)
        c4 = self.conv4_1(p3)
        c4 = self.conv4_2(c4)
        # (B, 128, 64, 96)
        p4 = self.pooling(c4)
        # (B, 128, 32, 48)
        c5 = self.conv5_1(p4)
        c5 = self.conv5_2(c5)
        # (B, 256, 32, 48)

        return c5