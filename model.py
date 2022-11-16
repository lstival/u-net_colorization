# Default imporations
from torch import nn
from torch.nn import Dropout, Softmax, Linear, LayerNorm, Conv2d

# Import network classes
from encoder import *
from vit import *

class ViT_UNet(nn.Module):
    def __init__(self, img_size=(512, 768), in_ch=3):
        super().__init__()

        self.pooling = nn.MaxPool2d(kernel_size=2)
        self.upsample = nn.Upsample(scale_factor=2)

        self.conv1_1 = CNNencoder_gn(in_ch, 16)
        self.conv1_2 = CNNencoder_gn(16, 16)
        self.conv2_1 = CNNencoder_gn(16, 32)
        self.conv2_2 = CNNencoder_gn(32, 32)
        self.conv3_1 = CNNencoder_gn(32, 64)
        self.conv3_2 = CNNencoder_gn(64, 64)
        self.conv4_1 = CNNencoder_gn(64, 128)
        self.conv4_2 = CNNencoder_gn(128, 128)
        self.conv5_1 = CNNencoder_gn(128, 256)
        self.conv5_2 = CNNencoder_gn(256, 256)

        self.vit = ViT(img_size)

        self.concat1 = Concat_gn(512, 128)
        self.convup1 = CNNencoder_gn(128, 128)
        self.concat2 = Concat_gn(256, 64)
        self.convup2 = CNNencoder_gn(64, 64)
        self.concat3 = Concat_gn(128, 32)
        self.convup3 = CNNencoder_gn(32, 32)
        self.concat4 = Concat_gn(64, 16)
        self.convup4 = CNNencoder_gn(16, 16)
        self.concat5 = Concat_ln(32, 23)
        self.convup5 = CNNencoder_ln(23, 23)

        self.Segmentation_head = nn.Conv2d(23, 3, kernel_size=1, stride=1, bias=False)
        self.sigmoid = nn.Sigmoid()


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
        # print(c5.shape)
        # (B, 256, 32, 48)
        v = self.vit(c5)
        # (B, 256, 16, 24)
        v1 = self.upsample(v)
        # (B, 256, 32, 48)
        u1 = self.concat1(v1, c5)
        u1 = self.convup1(u1)
        # (B, 128, 32, 48)
        u1 = self.upsample(u1)
        # (B, 128, 64, 96)
        u2 = self.concat2(u1, c4)
        u2 = self.convup2(u2)
        # (B, 64, 64, 96)
        u2 = self.upsample(u2)
        # (B, 64, 128, 192)
        u3 = self.concat3(u2, c3)
        u3 = self.convup3(u3)
        # (B, 32, 128, 192)
        u3 = self.upsample(u3)
        # (B, 32, 256, 384)
        u4 = self.concat4(u3, c2)
        u4 = self.convup4(u4)
        # (B, 16, 256, 384)
        u4 = self.upsample(u4)
        # (B, 16, 512, 768)
        u5 = self.concat5(u4, c1)
        u5 = self.convup5(u5)
        # (B, 23, 512, 768)
        out = self.Segmentation_head(u5)
        # (B, 23, 512, 768)

        sig = self.sigmoid(out)

        return sig