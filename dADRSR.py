import torch
import torch.nn as nn


def conv3x3(in_planes, out_planes, dilation):
    
    return nn.Conv2d(
        in_channels=in_planes,
        out_channels=out_planes,
        kernel_size=3,
        stride=1,
        padding='same',
        dilation=dilation,
        bias=False
     )

class SEBlock(nn.Module):
    def __init__(self, H, W, n_channels=30, r=10):
        super().__init__()
        self.n_channels = n_channels
        self.avgpool = nn.AvgPool2d((H, W))
        self.excitation = nn.Sequential(
            nn.Linear(n_channels, n_channels // r),
            nn.ReLU(True),
            nn.Linear(n_channels // r, n_channels)
        )
    def forward(self, x):
        input_tensor = x
        bs = x.shape[0]
        
        x = self.avgpool(x)
        x = x.view(bs, -1)
        x = self.excitation(x)
        X = torch.sigmoid(x)
        x = x.reshape(bs, self.n_channels, 1, 1)
        
        return input_tensor * x
    
class MARBBlock(nn.Module):
    def __init__(self, in_channels, H, W, n_filters_1=120, n_filters_2=10):
        super().__init__()
        self.branch_1 = nn.Sequential(
            conv3x3(in_channels, n_filters_1, dilation=1),
            nn.ReLU(True),
            conv3x3(n_filters_1, n_filters_2, dilation=1)
        )
        self.branch_2 = nn.Sequential(
            conv3x3(in_channels, n_filters_1, dilation=3),
            nn.ReLU(True),
            conv3x3(n_filters_1, n_filters_2, dilation=3)
        )
        self.branch_3 = nn.Sequential(
            conv3x3(in_channels, n_filters_1, dilation=5),
            nn.ReLU(True),
            conv3x3(n_filters_1, n_filters_2, dilation=5)
        )
        self.se = SEBlock(H, W)
    
    def forward(self, x):
        input_tensor = x
        
        x1 = self.branch_1(x)
        x2 = self.branch_2(x)
        x3 = self.branch_3(x)
        
        out = torch.cat([x1, x2, x3], axis=1)
        out = self.se(out)
        out += input_tensor
        
        return out
    
class dADR_SR(nn.Module):
    def __init__(self, in_channels, H, W, n_blocks=12, n_filters_1=120, n_filters_2=10):
        super().__init__()
        self.conv1_1 = conv3x3(in_channels, 10, dilation=1)
        self.conv1_2 = conv3x3(in_channels, 10, dilation=3)
        self.conv1_3 = conv3x3(in_channels, 10, dilation=5)
        self.blocks = self._make_layer(30, H, W, 120, 10, 12)
        self.conv2 = conv3x3(30, 30, dilation=1)
        self.conv3 = conv3x3(30, 1, dilation=1)
        
    def _make_layer(self, in_channels, H, W, n_filters_1, n_filters_2, blocks):
        layers = []
        for _ in range(blocks):
            layers.append(MARBBlock(in_channels, H, W, n_filters_1, n_filters_2))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x1 = self.conv1_1(x)
        x2 = self.conv1_2(x)
        x3 = self.conv1_3(x)
        x = torch.cat([x1, x2, x3], axis=1)
        
        out = self.blocks(x)
        out = self.conv2(out)
        out += x
        out = self.conv3(out)
        
        return out
