import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np
from torch.autograd import Variable


# class for upsample
class Upsample(nn.Module):
    def __init__(self, scale, act=False):
        super(Upsample, self).__init__()
        modules = []
        # add pixel shuffle
        modules.append(nn.PixelShuffle(scale))
        self.func = nn.Sequential(* modules)
        
    # forward
    def forward(self, x):
        out = self.func(x)
        return out

    
# make dense net
class Make_denseNet(nn.Module):
    def __init__(self, channels, growth_rate, kernel_size=3):
        '''
        channels: number of channels
        growth_rate: growth rate
        kernel_size: size of kernel
        '''
        super(Make_denseNet, self).__init__()
        self.conv = nn.Conv2d(channels, 
                       growth_rate, 
                       kernel_size=kernel_size,
                       padding=(kernel_size-1) // 2,
                       bias=False)
    
    def forward(self, x):
        '''
        forward
        x: input of dense net
        '''
        out = F.relu(self.conv(x))
        # cat x and out in 1 dimension
        out = torch.cat((x, out), 1)
        return out
    
    
# Residual Dense Block(RDB)
class RDB(nn.Module):
    def __init__(self, channels, nDenseLayers, growth_rate):
        '''
        channels: number of channels
        nDenseLayers: number of dense layers in a RDB
        growth_rate: growth rate
        '''
        super(RDB, self).__init__()
        channels_ = channels
        module = []
        
        # make n dense layer
        for i in range(nDenseLayers):
            module.append(Make_denseNet(channels_, growth_rate))
            channels_ += growth_rate
        
        # dense layers
        self.dense_layers = nn.Sequential(* module)
        # conv 1 x 1
        self.conv_1x1 = nn.Conv2d( channels_,
                          channels, 
                          kernel_size=1, 
                          padding=0, 
                          bias=False)
        
    def forward(self, x):
        '''
        forward
        x: input of a RDB
        '''
        out = self.dense_layers(x)
        out = self.conv_1x1(out)
        # add x and out
        out = out + x
        return out
        
        
# define Residual Dense Net(RDN)
class RDN(nn.Module):
    def __init__(self, channels=3, nDenseLayers=3, growth_rate=8, nFeat=32, scale=1):
        '''
        init of RDN:
        channels: number of channels
        nDenseLayers: number of dense layers in each RDB
        growth_rate: growth rate
        nFeat: channels of RDB input
        scale: scale
        '''
        super(RDN, self).__init__()
        nFeat_ = nFeat
        # F_-1
        self.F_1 = nn.Conv2d(channels, nFeat_, kernel_size=3, padding=1, bias=True)
        # F0
        self.F0 = nn.Conv2d(nFeat_, nFeat_, kernel_size=3, padding=1, bias=True)
        
        # 3 RDB
        self.RDB1 = RDB(nFeat_, nDenseLayers, growth_rate)
        self.RDB2 = RDB(nFeat_, nDenseLayers, growth_rate)
        self.RDB3 = RDB(nFeat_, nDenseLayers, growth_rate)
        
        # GFF: Global Feature Fusion
        self.GFF_1x1 = nn.Conv2d(nFeat_ * 3, nFeat_, kernel_size=1, padding=0, bias=True)
        self.GFF_3x3 = nn.Conv2d(nFeat_, nFeat_, kernel_size=3, padding=1, bias=True)
        
        # Upsample
        self.up_conv = nn.Conv2d(nFeat_, nFeat_ * scale * scale, kernel_size=3, padding=1, bias=True)
        self.up_sample = Upsample(scale)
        
        # last conv
        self.last_conv = nn.Conv2d(nFeat_, channels, kernel_size=3, padding=1, bias=True)
        
    def forward(self, x):
        '''
        forward
        x: input of RDN
        '''
        F_ = self.F_1(x)
        F0 = self.F0(F_)
        # RDB
        F1 = self.RDB1(F0)
        F2 = self.RDB2(F1)
        F3 = self.RDB3(F2)
        
        # cat F1, F2, F3 on dimension 1
        cat_out = torch.cat((F1, F2, F3), 1)
        # GFF
        F_dLF = self.GFF_1x1(cat_out)
        F_GF = self.GFF_3x3(F_dLF)
        
        # add F_GF and F_
        F_DF = F_GF + F_
        
        # up conv and up smaple
        up_con = self.up_conv(F_DF)
        up_sam = self.up_sample(up_con)
        
        out = self.last_conv(up_sam)
        
        return out
        