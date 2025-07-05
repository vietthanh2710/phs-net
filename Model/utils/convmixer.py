import torch
import torch.nn as nn

class ConvMixerBlock(nn.Module):
    def __init__(self, filters_in, filters_out, kernel_size):
        super(ConvMixerBlock, self).__init__()
        self.depthwise_conv = nn.Conv2d(
            filters_in,
            filters_in,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=filters_in,
        )
        self.pointwise_conv = nn.Conv2d(filters_in, filters_out, kernel_size=1)
        self.scale = nn.Parameter(torch.ones(1))

        self.act = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(filters_in)
        self.bn2 = nn.BatchNorm2d(filters_out)
    def forward(self, x):
        x0 = x
        x = self.depthwise_conv(x)
        x = self.bn1(self.act(x + self.scale * x0))
        x = self.pointwise_conv(x)
        x = self.act(x)
        x = self.bn2(x)
        return x