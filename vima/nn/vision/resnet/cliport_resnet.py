import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ["CLIPortResNet"]


class IdentityBlock(nn.Module):
    def __init__(
        self, in_planes, filters, kernel_size, stride=1, final_relu=True, batchnorm=True
    ):
        super().__init__()
        self.final_relu = final_relu
        self.batchnorm = batchnorm

        filters1, filters2, filters3 = filters
        self.conv1 = nn.Conv2d(in_planes, filters1, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(filters1) if self.batchnorm else nn.Identity()
        self.conv2 = nn.Conv2d(
            filters1,
            filters2,
            kernel_size=kernel_size,
            dilation=1,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(filters2) if self.batchnorm else nn.Identity()
        self.conv3 = nn.Conv2d(filters2, filters3, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(filters3) if self.batchnorm else nn.Identity()

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += x
        if self.final_relu:
            out = F.relu(out)
        return out


class ConvBlock(nn.Module):
    def __init__(
        self, in_planes, filters, kernel_size, stride=1, final_relu=True, batchnorm=True
    ):
        super(ConvBlock, self).__init__()
        self.final_relu = final_relu
        self.batchnorm = batchnorm

        filters1, filters2, filters3 = filters
        self.conv1 = nn.Conv2d(in_planes, filters1, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(filters1) if self.batchnorm else nn.Identity()
        self.conv2 = nn.Conv2d(
            filters1,
            filters2,
            kernel_size=kernel_size,
            dilation=1,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(filters2) if self.batchnorm else nn.Identity()
        self.conv3 = nn.Conv2d(filters2, filters3, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(filters3) if self.batchnorm else nn.Identity()

        self.shortcut = nn.Sequential(
            nn.Conv2d(in_planes, filters3, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(filters3) if self.batchnorm else nn.Identity(),
        )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        if self.final_relu:
            out = F.relu(out)
        return out


class CLIPortResNet(nn.Module):
    def __init__(self, in_channels: int, output_dim: int, batch_norm: bool):
        super().__init__()
        self.in_chan = in_channels
        self.batchnorm = batch_norm

        self.layers = self._make_layers()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, output_dim)

    def _make_layers(self):
        layers = nn.Sequential(
            # conv1
            nn.Conv2d(self.in_chan, 64, stride=1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64) if self.batchnorm else nn.Identity(),
            nn.ReLU(True),
            # fcn
            ConvBlock(
                64, [64, 64, 64], kernel_size=3, stride=1, batchnorm=self.batchnorm
            ),
            IdentityBlock(
                64, [64, 64, 64], kernel_size=3, stride=1, batchnorm=self.batchnorm
            ),
            ConvBlock(
                64, [128, 128, 128], kernel_size=3, stride=2, batchnorm=self.batchnorm
            ),
            IdentityBlock(
                128, [128, 128, 128], kernel_size=3, stride=1, batchnorm=self.batchnorm
            ),
            ConvBlock(
                128, [256, 256, 256], kernel_size=3, stride=2, batchnorm=self.batchnorm
            ),
            IdentityBlock(
                256, [256, 256, 256], kernel_size=3, stride=1, batchnorm=self.batchnorm
            ),
            ConvBlock(
                256, [512, 512, 512], kernel_size=3, stride=2, batchnorm=self.batchnorm
            ),
            IdentityBlock(
                512, [512, 512, 512], kernel_size=3, stride=1, batchnorm=self.batchnorm
            ),
            # head
            ConvBlock(
                512, [256, 256, 256], kernel_size=3, stride=1, batchnorm=self.batchnorm
            ),
            IdentityBlock(
                256, [256, 256, 256], kernel_size=3, stride=1, batchnorm=self.batchnorm
            ),
            nn.UpsamplingBilinear2d(scale_factor=2),
            ConvBlock(
                256, [128, 128, 128], kernel_size=3, stride=1, batchnorm=self.batchnorm
            ),
            IdentityBlock(
                128, [128, 128, 128], kernel_size=3, stride=1, batchnorm=self.batchnorm
            ),
            nn.UpsamplingBilinear2d(scale_factor=2),
            ConvBlock(
                128, [64, 64, 64], kernel_size=3, stride=1, batchnorm=self.batchnorm
            ),
            IdentityBlock(
                64, [64, 64, 64], kernel_size=3, stride=1, batchnorm=self.batchnorm
            ),
            nn.UpsamplingBilinear2d(scale_factor=2),
            # conv2
            ConvBlock(
                64,
                [16, 16, 512],
                kernel_size=3,
                stride=1,
                final_relu=False,
                batchnorm=self.batchnorm,
            ),
            IdentityBlock(
                512,
                [16, 16, 512],
                kernel_size=3,
                stride=1,
                final_relu=False,
                batchnorm=self.batchnorm,
            ),
        )
        return layers

    def forward(self, x):
        x = self.layers(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x
