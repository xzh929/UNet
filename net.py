import torch
from torch import nn
from torch.nn.functional import interpolate


class Conv_Block(nn.Module):
    def __init__(self, c_in, c_out):
        super(Conv_Block, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, 1, padding=1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(c_out),
            nn.LeakyReLU(),
            nn.Conv2d(c_out, c_out, 3, 1, padding=1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(c_out),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.layer(x)


class DownSample(nn.Module):
    def __init__(self, c):
        super(DownSample, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(c, c, 3, 2, 1, padding_mode='reflect'),  # 卷积下采样
            nn.BatchNorm2d(c),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.layer(x)


class UpSample(nn.Module):
    def __init__(self, c):
        super(UpSample, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(c, c // 2, 3, 1, 1, padding_mode='reflect'),
            nn.BatchNorm2d(c // 2),
            nn.LeakyReLU()
        )

    def forward(self, x, r):
        up = interpolate(x, scale_factor=2)  # 放大2倍
        out = self.layer(up)
        return torch.cat((out, r), 1)


class Unet(nn.Module):
    def __init__(self):
        super(Unet, self).__init__()
        self.Conv1 = Conv_Block(3, 64)
        self.down1 = DownSample(64)
        self.Conv2 = Conv_Block(64, 128)
        self.down2 = DownSample(128)
        self.Conv3 = Conv_Block(128, 256)
        self.down3 = DownSample(256)
        self.Conv4 = Conv_Block(256, 512)
        self.down4 = DownSample(512)
        self.Conv5 = Conv_Block(512, 1024)
        self.up1 = UpSample(1024)
        self.Conv6 = Conv_Block(1024, 512)
        self.up2 = UpSample(512)
        self.Conv7 = Conv_Block(512, 256)
        self.up3 = UpSample(256)
        self.Conv8 = Conv_Block(256, 128)
        self.up4 = UpSample(128)
        self.Conv9 = Conv_Block(128, 64)
        self.pre = nn.Conv2d(64, 3, 3, 1, 1)

    def forward(self, x):
        R1 = self.Conv1(x)
        R2 = self.Conv2(self.down1(R1))
        R3 = self.Conv3(self.down2(R2))
        R4 = self.Conv4(self.down3(R3))
        Y1 = self.Conv5(self.down4(R4))
        U1 = self.Conv6(self.up1(Y1, R4))
        U2 = self.Conv7(self.up2(U1, R3))
        U3 = self.Conv8(self.up3(U2, R2))
        U4 = self.Conv9(self.up4(U3, R1))
        return self.pre(U4)


if __name__ == '__main__':
    a = torch.randn(1,3,256,256)
    net = Unet()
    out = net(a)
    print(a.shape)