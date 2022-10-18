import torch
import torch.nn as nn
import numpy as np
from .downsampler import Downsampler

# attention module for FRU
class attention_FRU(nn.Module):
    def __init__(self, num_channels_down, pad='reflect'):
        super(attention_FRU, self).__init__()
        # layers to generate conditional convolution weights
        self.gen_se_weights1 = nn.Sequential(
            nn.Conv2d(num_channels_down, num_channels_down, 1, padding_mode=pad),
            # nn.LeakyReLU(0.1, inplace=True),
            nn.Softplus(),
            nn.Sigmoid())

        # create conv layers
        self.conv_1 = nn.Conv2d(num_channels_down, num_channels_down, 1, padding_mode=pad)
        self.norm_1 = nn.BatchNorm2d(num_channels_down, affine=False)
        self.actvn = nn.LeakyReLU(0.2, inplace=True)
        # self.actvn = nn.Softplus()

    def forward(self, guide, x):
        se_weights1 = self.gen_se_weights1(guide)
        dx = self.conv_1(x)
        dx = self.norm_1(dx)
        dx = torch.mul(dx, se_weights1)
        out = self.actvn(dx)
        return out

# attention module for URU
class attention_URU(nn.Module):
    def __init__(self, num_channels_down, pad='reflect', upsample_mode='bilinear',scale=2):
        super(attention_URU, self).__init__()

        # generate conditional convolution weights
        self.weight_map = nn.Sequential(
            nn.Conv2d(num_channels_down, num_channels_down, 1, padding_mode=pad),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Sigmoid())

        # upsampling and channel-wise normalization
        self.upsample_norm = nn.Sequential(
            nn.Upsample(scale_factor=scale, mode=upsample_mode),
            nn.BatchNorm2d(num_channels_down, affine=False))

    def forward(self, guide, x):
        x_upsample = self.upsample_norm(x)
        weight = self.weight_map(guide)
        out = torch.mul(x_upsample, weight)
        return out

# guided deep decoder
# class s2_net(nn.Module):
#     def __init__(self, in_channels=12, out_channels=12, num_channels_down=64, num_channels_up=64, num_channels_skip=4,
#                  filter_size_down=3, filter_size_up=3, filter_skip_size=1):
#         super().__init__()
#
#         self.FRU = attention_FRU(num_channels_down)
#         # self.URU1 = nn.Upsample(scale_factor=3,mode='bicubic')
#         # self.URU2 = nn.Upsample(scale_factor=2,mode='bicubic')
#
#         self.guide_20 = nn.Sequential(
#             nn.Conv2d(y20_channel, num_channels_down, filter_size_down, padding ='same',padding_mode='reflect'),
#             nn.BatchNorm2d(num_channels_down),
#             nn.LeakyReLU(0.2, inplace=True))
#
#         self.guide_10 = nn.Sequential(
#             nn.Conv2d(y10_channel, num_channels_down, filter_size_down, padding = 'same',padding_mode='reflect'),
#             nn.BatchNorm2d(num_channels_down),
#             nn.LeakyReLU(0.2, inplace=True))
#
#         self.enc60 = nn.Sequential(
#             nn.Conv2d(y20_channel+y60_channel, num_channels_down, filter_size_down,padding='same', padding_mode='reflect'),
#             nn.BatchNorm2d(num_channels_down),
#             nn.LeakyReLU(0.2, inplace=True))
#
#         self.enc20 = nn.Sequential(
#             nn.Conv2d(y20_channel+y10_channel+num_channels_down, num_channels_down, filter_size_down,
#                       padding='same',padding_mode='reflect'),
#             nn.BatchNorm2d(num_channels_down),
#             nn.LeakyReLU(0.2, inplace=True))
#
#         self.enc = nn.Sequential(
#             nn.Conv2d(num_channels_down, num_channels_down, filter_size_down,padding='same', padding_mode='reflect'),
#             nn.BatchNorm2d(num_channels_down),
#             nn.LeakyReLU(0.2, inplace=True))
#
#         self.skip = nn.Sequential(
#             nn.Conv2d(num_channels_down, num_channels_skip, filter_skip_size, padding ='same', padding_mode='reflect'),
#             nn.BatchNorm2d(num_channels_skip),
#             nn.LeakyReLU(0.2, inplace=True))
#
#         self.dc = nn.Sequential(
#             nn.Conv2d((num_channels_skip + num_channels_up), num_channels_up, filter_size_up,padding='same',padding_mode='reflect'),
#             nn.BatchNorm2d(num_channels_up),
#             nn.LeakyReLU(0.2, inplace=True))
#         self.out_layer = nn.Sequential(
#             nn.Conv2d(num_channels_up, 12, 1, padding_mode='reflect'),
#             nn.Sigmoid())
#         self.conv60 = nn.Conv2d(y60_channel,num_channels_down,filter_size_down, padding = 'same',padding_mode = 'reflect')
#         self.conv20 = nn.Conv2d(num_channels_down+y20_channel+y10_channel,num_channels_down,filter_size_down,
#                                 padding = 'same',padding_mode = 'reflect')
#         self.conv = nn.Conv2d(num_channels_down,num_channels_down,filter_size_down, padding = 'same',padding_mode = 'reflect')
#     def forward(self, inputs):
#         y10 = inputs[:,:4,:,:]
#         y20 = inputs[:,4:10,:,:]
#         y60 = inputs[:,10:,:,:]
#         y20_en0 = self.guide_20(y20)
#         y20_en1 = self.enc(y20_en0)
#         y20_en2 = self.enc(y20_en1)
#         y20_en3 = self.enc(y20_en2)
#         y20_en4 = self.enc(y20_en3)
#
#         y20_dc0 = self.enc(y20_en4)
#         y20_dc1 = self.dc(torch.cat((self.skip(y20_en4), y20_dc0), dim=1))
#         y20_dc2 = self.dc(torch.cat((self.skip(y20_en3), y20_dc1), dim=1))
#         y20_dc3 = self.dc(torch.cat((self.skip(y20_en2), y20_dc2), dim=1))
#         y20_dc4 = self.dc(torch.cat((self.skip(y20_en1), y20_dc3), dim=1))
#         y20_dc5 = self.dc(torch.cat((self.skip(y20_en0), y20_dc4), dim=1))
#
#         # x60_1 = self.URU1(y60) #
#         x60_1 = self.FRU(y20_dc1, self.conv60(y60))
#         x60_2 = self.FRU(y20_dc2, self.conv(x60_1))
#         x60_3 = self.FRU(y20_dc3, self.conv(x60_2))
#         x60_4 = self.FRU(y20_dc4, self.conv(x60_3))
#         x60_5 = self.FRU(y20_dc5, self.conv(x60_4))
#         # For 20 m bands
#         y10_en0 = self.guide_10(y10)
#         y10_en1 = self.enc(y10_en0)
#         y10_en2 = self.enc(y10_en1)
#         y10_en3 = self.enc(y10_en2)
#         y10_en4 = self.enc(y10_en3)
#
#         y10_dc0 = self.enc(y10_en4)
#         y10_dc1 = self.dc(torch.cat((self.skip(y10_en4), y10_dc0), dim=1))
#         y10_dc2 = self.dc(torch.cat((self.skip(y10_en3), y10_dc1), dim=1))
#         y10_dc3 = self.dc(torch.cat((self.skip(y10_en2), y10_dc2), dim=1))
#         y10_dc4 = self.dc(torch.cat((self.skip(y10_en1), y10_dc3), dim=1))
#         y10_dc5 = self.dc(torch.cat((self.skip(y10_en0), y10_dc4), dim=1))
#
#         # xout_11 = self.URU2(y20) #c=6
#         # xout_12 = self.URU2(x60_5) #c=64
#         xout1 = torch.cat((x60_5,y20,y10),dim=1) #c=64+6+2
#         xout1 = self.FRU(y10_dc1, self.conv20(xout1))
#         xout2 = self.FRU(y10_dc2, self.conv(xout1))
#         xout3 = self.FRU(y10_dc3, self.conv(xout2))
#         xout4 = self.FRU(y10_dc4, self.conv(xout3))
#         xout5 = self.FRU(y10_dc5, self.conv(xout4))
#
#         out = self.out_layer(xout5)
#
#         return out

class s2_attention_net(nn.Module):
    def __init__(self, g_channel=4, in_channel=8, out_channel=12, num_channels_down=64, num_channels_up=64, num_channels_skip=32,
                 filter_size_down=3, filter_size_up=3, filter_skip_size=1):
        super().__init__()
        self.FRU = attention_FRU(num_channels_down)
        self.enc0 = nn.Sequential(
            nn.Conv2d(g_channel, num_channels_down, filter_size_down,
                      padding='same',padding_mode='reflect'),
            nn.BatchNorm2d(num_channels_down),
            nn.LeakyReLU(0.2, inplace=True))
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channel, num_channels_down, filter_size_down,
                      padding='same', padding_mode='reflect'),
            nn.BatchNorm2d(num_channels_down),
            nn.LeakyReLU(0.2, inplace=True))
        self.enc = nn.Sequential(
            nn.Conv2d(num_channels_down, num_channels_down, filter_size_down,padding='same', padding_mode='reflect'),
            nn.BatchNorm2d(num_channels_down),
            nn.LeakyReLU(0.2, inplace=True))

        self.skip = nn.Sequential(
            nn.Conv2d(num_channels_down, num_channels_skip, filter_skip_size, padding ='same', padding_mode='reflect'),
            nn.BatchNorm2d(num_channels_skip),
            nn.LeakyReLU(0.2, inplace=True))

        self.dc = nn.Sequential(
            nn.Conv2d((num_channels_skip + num_channels_up), num_channels_up, filter_size_up,padding='same',padding_mode='reflect'),
            nn.BatchNorm2d(num_channels_up),
            nn.LeakyReLU(0.2, inplace=True))
        self.out_layer = nn.Sequential(
            nn.Conv2d(num_channels_up, out_channel, 1, padding_mode='reflect'),
            nn.Sigmoid())
        self.conv = nn.Conv2d(num_channels_down,num_channels_down,filter_size_down, padding = 'same',padding_mode = 'reflect')
    def forward(self, inputs):
        '''inputs: concat(y10,y20up,y60up)'''
        # encoder part
        yg = inputs[:,:4,:,:]
        yin = inputs[:, 4:, :, :]
        y_en0 = self.enc0(yg)
        y_en1 = self.enc(y_en0)
        y_en2 = self.enc(y_en1)
        y_en3 = self.enc(y_en2)
        y_en4 = self.enc(y_en3)
        # decoder part with skip connections
        y_dc0 = self.enc(y_en4)
        y_dc1 = self.dc(torch.cat((self.skip(y_en4), y_dc0), dim=1))
        y_dc2 = self.dc(torch.cat((self.skip(y_en3), y_dc1), dim=1))
        y_dc3 = self.dc(torch.cat((self.skip(y_en2), y_dc2), dim=1))
        y_dc4 = self.dc(torch.cat((self.skip(y_en1), y_dc3), dim=1))
        y_dc5 = self.dc(torch.cat((self.skip(y_en0), y_dc4), dim=1))

        xout1 = self.enc1(yin)  # c=64+6+2
        xout1 = self.FRU(y_dc1, self.conv(xout1))
        xout2 = self.FRU(y_dc2, self.conv(xout1))
        xout3 = self.FRU(y_dc3, self.conv(xout2))
        xout4 = self.FRU(y_dc4, self.conv(xout3))
        xout5 = self.FRU(y_dc5, self.conv(xout4))

        out = self.out_layer(xout5)

        return out
class s2_net(nn.Module):
    def __init__(self, in_channel=12, out_channel=12, num_channels_down=64, num_channels_up=64, num_channels_skip=4,
                 filter_size_down=3, filter_size_up=3, filter_skip_size=1):
        super().__init__()
        self.enc0 = nn.Sequential(
            nn.Conv2d(in_channel, num_channels_down, filter_size_down,
                      padding='same',padding_mode='reflect'),
            nn.BatchNorm2d(num_channels_down),
            nn.LeakyReLU(0.2, inplace=True))
        self.enc = nn.Sequential(
            nn.Conv2d(num_channels_down, num_channels_down, filter_size_down,padding='same', padding_mode='reflect'),
            nn.BatchNorm2d(num_channels_down),
            nn.LeakyReLU(0.2, inplace=True))

        self.skip = nn.Sequential(
            nn.Conv2d(num_channels_down, num_channels_skip, filter_skip_size, padding ='same', padding_mode='reflect'),
            nn.BatchNorm2d(num_channels_skip),
            nn.LeakyReLU(0.2, inplace=True))

        self.dc = nn.Sequential(
            nn.Conv2d((num_channels_skip + num_channels_up), num_channels_up, filter_size_up,padding='same',padding_mode='reflect'),
            nn.BatchNorm2d(num_channels_up),
            nn.LeakyReLU(0.2, inplace=True))
        self.out_layer = nn.Sequential(
            nn.Conv2d(num_channels_up, out_channel, 1, padding_mode='reflect'),
            nn.Sigmoid())
        self.conv = nn.Conv2d(num_channels_down,num_channels_down,filter_size_down, padding = 'same',padding_mode = 'reflect')
    def forward(self, inputs):
        '''inputs: concat(y10,y20up,y60up)'''
        # encoder part
        y_en0 = self.enc0(inputs)
        y_en1 = self.enc(y_en0)
        y_en2 = self.enc(y_en1)
        y_en3 = self.enc(y_en2)
        y_en4 = self.enc(y_en3)
        # decoder part with skip connections
        y_dc0 = self.enc(y_en4)
        y_dc1 = self.dc(torch.cat((self.skip(y_en4), y_dc0), dim=1))
        y_dc2 = self.dc(torch.cat((self.skip(y_en3), y_dc1), dim=1))
        y_dc3 = self.dc(torch.cat((self.skip(y_en2), y_dc2), dim=1))
        y_dc4 = self.dc(torch.cat((self.skip(y_en1), y_dc3), dim=1))
        y_dc5 = self.dc(torch.cat((self.skip(y_en0), y_dc4), dim=1))
        out = self.out_layer(y_dc5)

        return out
