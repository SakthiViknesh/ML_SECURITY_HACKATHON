"""Surrogate CNN architectures used to craft transfer-based adversarial examples.

These are NOT the target black-box model -- they are independently designed/trained
networks used only to compute gradients for generating candidate perturbations. The
target model is only ever queried for its predicted label.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, c_in, c_out, pool=False):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(c_out)
        self.pool = pool

    def forward(self, x):
        x = F.relu(self.bn(self.conv(x)))
        if self.pool:
            x = F.max_pool2d(x, 2)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c1 = ConvBlock(c, c)
        self.c2 = ConvBlock(c, c)

    def forward(self, x):
        return x + self.c2(self.c1(x))


class ResNet9(nn.Module):
    """Small ResNet-9 style network (res9 architecture popular for fast CIFAR-10 training)."""

    def __init__(self, num_classes=10):
        super().__init__()
        self.stem = ConvBlock(3, 64)
        self.l1 = ConvBlock(64, 128, pool=True)
        self.r1 = ResidualBlock(128)
        self.l2 = ConvBlock(128, 256, pool=True)
        self.l3 = ConvBlock(256, 512, pool=True)
        self.r2 = ResidualBlock(512)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.l1(x)
        x = self.r1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.r2(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)


class VGGSmall(nn.Module):
    """Plain (non-residual) VGG-style CNN -- architecturally distinct from ResNet9,
    used as the second member of the surrogate ensemble to diversify gradients."""

    def __init__(self, num_classes=10):
        super().__init__()
        def block(c_in, c_out, pool=True):
            layers = [nn.Conv2d(c_in, c_out, 3, padding=1), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True)]
            if pool:
                layers.append(nn.MaxPool2d(2))
            return nn.Sequential(*layers)

        self.features = nn.Sequential(
            block(3, 64, pool=False),
            block(64, 64, pool=True),
            block(64, 128, pool=False),
            block(128, 128, pool=True),
            block(128, 256, pool=False),
            block(256, 256, pool=True),
            block(256, 256, pool=False),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).flatten(1)
        x = self.dropout(x)
        return self.fc(x)
