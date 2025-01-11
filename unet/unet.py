import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    '''双卷积'''
 
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True)
        )
 
    def forward(self, x):
        return self.double_conv(x)


class InConv(nn.Module):
    '''输入层'''
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(InConv, self).__init__()
        self.in_conv = DoubleConv(in_channels, out_channels, kernel_size, padding)

    def forward(self, x):
        return self.in_conv(x)


class DownSampling(nn.Module):
    '''下采样'''
 
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(DownSampling, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            DoubleConv(in_channels, out_channels, kernel_size, padding)
        )
 
    def forward(self, x):
        return self.maxpool_conv(x)
 
 
class UpSampling(nn.Module):
    '''上采样'''
 
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(UpSampling, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.double_conv = DoubleConv(in_channels, out_channels, kernel_size, padding)
 
    def forward(self, x1, x2):
        x1 = self.up(x1)

        # 调整x2的尺寸，使其与x1的尺寸相同
        # x2 = F.interpolate(x2, size=x1.shape[-2:], mode="bilinear", align_corners=True)

        # 调整x1的尺寸，使其与x2的尺寸相同
        x1 = F.interpolate(x1, size=x2.shape[-2:], mode="bilinear", align_corners=True)
 
        x = torch.cat((x2, x1), dim=1)
        return self.double_conv(x)
 
 
class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
 
    def forward(self, x):
        return self.conv(x)
    



class UNet(nn.Module):
    '''U-Net'''
    
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(UNet, self).__init__()

        self.in_conv = InConv(in_channels, 64, kernel_size, padding)
        self.down1 = DownSampling(64, 128, kernel_size, padding)
        self.down2 = DownSampling(128, 256, kernel_size, padding)
        self.down3 = DownSampling(256, 512, kernel_size, padding)
        self.down4 = DownSampling(512, 1024, kernel_size, padding)
        self.up1 = UpSampling(1024, 512, kernel_size, padding)
        self.up2 = UpSampling(512, 256, kernel_size, padding)
        self.up3 = UpSampling(256, 128, kernel_size, padding)
        self.up4 = UpSampling(128, 64, kernel_size, padding)
        self.out_conv = OutConv(64, out_channels)

    def forward(self, x):
        self.x1 = self.in_conv(x)
        self.x2 = self.down1(self.x1)
        self.x3 = self.down2(self.x2)
        self.x4 = self.down3(self.x3)
        self.x5 = self.down4(self.x4)
        self.x6 = self.up1(self.x5, self.x4)
        self.x7 = self.up2(self.x6, self.x3)
        self.x8 = self.up3(self.x7, self.x2)
        self.x9 = self.up4(self.x8, self.x1)
        output  = self.out_conv(self.x9)
        # 将输出调整为原始尺寸
        # output = F.interpolate(output, size=(config.IMG_SIZE, config.IMG_SIZE), mode='bilinear', align_corners=True)
        return output
