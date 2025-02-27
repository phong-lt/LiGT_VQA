"""
This is the adaptation of U-Net and T5+U

Reference: 
    github: - https://github.com/due-benchmark
            - https://github.com/milesial/Pytorch-UNet
            - https://github.com/uakarsh/TiLT-Implementation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import roi_pool 


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels=3, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.bilinear = bilinear

        self.inc1 = (DoubleConv(n_channels, 128))
        self.scaler = (nn.MaxPool2d(kernel_size=8, stride=8))
        self.inc2 = (DoubleConv(128, 128))

        self.down1 = (Down(128, 256))
        self.down2 = (Down(256, 512))
        self.down3 = (Down(512, 1024))

        factor = 2 if bilinear else 1

        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128, bilinear))


    def forward(self, x):
        x1 = self.inc1(x)
        x1 = self.scaler(x1)
        x1 = self.inc2(x1)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
       
        return x

    def use_checkpointing(self):
        self.inc1 = torch.utils.checkpoint(self.inc1)
        self.scaler = torch.utils.checkpoint(self.scaler)
        self.inc2 = torch.utils.checkpoint(self.inc2)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)



class RoIPool(nn.Module):

    def __init__(self, output_size=(3, 3), spatial_scale=0.125):
        super().__init__()
        self.output_size = output_size
        self.spatial_scale = spatial_scale
        self.roi_pool = roi_pool

    def forward(self, image_embedding, bboxes):

        feature_maps_bboxes = []
        for single_batch_img, single_batch_bbox in zip(image_embedding, bboxes):
            feature_map_single_batch = self.roi_pool(input=single_batch_img.unsqueeze(0),
                                                     boxes=torch.cat([torch.zeros(single_batch_bbox.shape[0], 1).to(
                                                         single_batch_bbox.device), single_batch_bbox], axis=-1).float(),
                                                     output_size=self.output_size,
                                                     spatial_scale = self.spatial_scale
                                                     )
            feature_maps_bboxes.append(feature_map_single_batch)

        return torch.stack(feature_maps_bboxes, axis=0) 