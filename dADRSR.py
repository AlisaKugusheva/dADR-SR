import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


def SE_block(input_tensor, r=10):
    x = input_tensor
    bs = x.shape[0]
    n_channels = x.shape[1]
    
    x = nn.AvgPool2d((x.shape[2], x.shape[3]))(x)
    x = x.view(bs, n_channels)
    x = nn.Linear(n_channels, n_channels // r)(x)
    x = nn.ReLU()(x)
    x = nn.Linear(n_channels // r, n_channels)(x)
    x = torch.sigmoid(x)
    x = x.reshape(bs, n_channels, 1, 1)
    
    return input_tensor * x


def M_ARB(input_tensor, n_filters_1=120, n_filters_2=10):
    x = input_tensor
    n_channels = x.shape[1]
    
    x1 = nn.Conv2d(n_channels, n_filters_1, kernel_size=3, stride=1, padding='same', dilation=1)(x)
    x1 = nn.ReLU()(x1)
    x1 = nn.Conv2d(n_filters_1, n_filters_2, kernel_size=3, stride=1, padding='same', dilation=1)(x1)
    
    x2 = nn.Conv2d(n_channels, n_filters_1, kernel_size=3, stride=1, padding='same', dilation=3)(x)
    x2 = nn.ReLU()(x2)
    x2 = nn.Conv2d(n_filters_1, n_filters_2, kernel_size=3, stride=1, padding='same', dilation=3)(x2)
    
    x3 = nn.Conv2d(n_channels, n_filters_1, kernel_size=3, stride=1, padding='same', dilation=5)(x)
    x3 = nn.ReLU()(x3)
    x3 = nn.Conv2d(n_filters_1, n_filters_2, kernel_size=3, stride=1, padding='same', dilation=5)(x3)
    
    x = torch.cat([x1, x2, x3], axis=1)
    x = SE_block(x)
    
    return x + input_tensor


class dADR_SR(nn.Module):
    def __init__(self, n_channels=4, n_filters=10, n_blocks=12, n_filters_1=120, n_filters_2=10):
        super().__init__()
        self.n_channels = n_channels
        self.n_filters = n_filters
        self.n_blocks = n_blocks
        self.n_filters_1 = n_filters_1
        self.n_filters_2 = n_filters_2
        
    def forward(self, x):       
        x1 = nn.Conv2d(in_channels=self.n_channels, 
                       out_channels=self.n_filters, 
                       kernel_size=3, 
                       stride=1, 
                       padding='same', 
                       dilation=1)(x)
        
        x2 = nn.Conv2d(in_channels=self.n_channels, 
                       out_channels=self.n_filters, 
                       kernel_size=3, 
                       stride=1, 
                       padding='same', 
                       dilation=3)(x)
        
        x3 = nn.Conv2d(in_channels=self.n_channels, 
                       out_channels=self.n_filters, 
                       kernel_size=3, 
                       stride=1, 
                       padding='same', 
                       dilation=5)(x)
        
        x = x_inp = torch.cat([x1, x2, x3], axis=1)
        
        for _ in range(self.n_blocks):
            x = M_ARB(x, n_filters_1=self.n_filters_1, n_filters_2=self.n_filters_2)

        x = nn.Conv2d(in_channels=x.shape[1],
                      out_channels=x.shape[1],
                      kernel_size=3,
                      padding='same')(x)
        
        x = x_inp + x
        x = nn.Conv2d(in_channels=x.shape[1],
                      out_channels=1,
                      kernel_size=3,
                      padding='same')(x)
        
        return x