import torch
import torch.nn as nn

class SpatialAttention(nn.Module):

    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, 1, kernel_size=7,padding=3), 

            nn.ReLU(),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return x * self.conv(x)  

class ChannelAttention(nn.Module):

    def __init__(self, channels, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels//reduction, 1),
            nn.Conv2d(channels//reduction, channels, 1),

            nn.Sigmoid()
        )
    
    def forward(self, x):
        return x * self.conv(self.avg_pool(x))  

class DWCBR(nn.Module):

    def __init__(self, in_c, out_c):
        super().__init__()
        self.dwconv = nn.Sequential(
            nn.Conv2d(in_c, in_c, 3, padding=1, groups=in_c),
            nn.BatchNorm2d(in_c),
            nn.ReLU(),
            nn.Conv2d(in_c, out_c, 1),
            nn.BatchNorm2d(out_c),
            nn.ReLU()
        )
    
    def forward(self, x):
        return self.dwconv(x)

class AIRBlock(nn.Module):

    def __init__(self, in_channels, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        out_c = in_channels // 2 

        self.input_path = nn.Sequential(
            nn.Conv2d(in_channels, out_c, 1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(),
            DWCBR(out_c, out_c),              
            nn.Conv2d(out_c, out_c, 1),     
            nn.BatchNorm2d(out_c)
        )

        self.split_conv = nn.Conv2d(out_c, out_c*3, 1)  

        self.so = SpatialOperation(out_c)
        self.co = ChannelOperation(out_c)

        self.dw_conv1 = nn.Sequential(
            nn.Conv2d(out_c, out_c, 3, padding=1, groups=out_c),
            nn.BatchNorm2d(out_c)
        )
        self.dw_conv2 = nn.Sequential(
            nn.Conv2d(out_c, out_c, 3, padding=1, groups=out_c),
            nn.BatchNorm2d(out_c)
        )

        self.output_conv = nn.Sequential(
            nn.Conv2d(out_c, c2, 1),  
            nn.BatchNorm2d(c2),
            nn.ReLU()
        )
        self.r = nn.Conv2d(in_channels, c2, 1)
    def forward(self, x):

        x1 = self.input_path(x)      

        split = self.split_conv(x1)    
        b1, b2, b3 = torch.split(split, split.shape[1]//3, dim=1)

        b1 = self.co(self.so(b1))  
        b2 = self.so(self.co(b2))  
        merged = b1 + b2          

        merged = self.dw_conv1(merged) 
        merged = merged * b3      
        merged = self.dw_conv2(merged) 

        return self.output_conv(merged) + self.r(x) 


class DPDFBlock(nn.Module):
    def __init__(self, in_channels, out_channels, k=1, s=1, p=None, g=1, d=1, act=True):
        super(EAP, self).__init__()
        self.dw_cbr = DWCBR(in_channels, out_channels)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.spatial_attention = SpatialOperation(out_channels) 
        self.channel_attention = ChannelOperation(out_channels)  
        self.cbr_1x1 = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x_dw = self.dw_cbr(x)

        max_pool_out = self.max_pool(x_dw)
        max_pool_out = self.spatial_attention(max_pool_out)

        avg_pool_out = self.avg_pool(x_dw)
        avg_pool_out = self.channel_attention(avg_pool_out)

        out = max_pool_out + avg_pool_out

        out = self.cbr_1x1(out)
        
        return out