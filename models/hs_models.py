import torch
import torch.nn as nn

# attention module for FRU
class attention_FRU(nn.Module):
    def __init__(self, num_channels_down, pad='reflect'):
        super(attention_FRU, self).__init__()
        # layers to generate conditional convolution weights
        self.gen_se_weights1 = nn.Sequential(
            nn.Conv2d(num_channels_down, num_channels_down, 1, padding_mode=pad),
            nn.LeakyReLU(0.2, inplace=True), # Dont use Softplus here
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

# guided deep decoder
class hs_net(nn.Module):
    def __init__(self, ym_channel=1, yh_channel=93, num_channels_down=128, num_channels_up=128, num_channels_skip=64,
                 filter_size_down=3, filter_size_up=3, filter_skip_size=1):
        super(hs_net,self).__init__()

        self.FRU = attention_FRU(num_channels_down)
        self.up_bic = nn.Upsample(scale_factor=4, mode='bicubic')
        self.up_trans = nn.ConvTranspose2d(yh_channel,yh_channel,filter_size_down,stride=4,padding=1)

        self.guide_ms = nn.Sequential(
            nn.Conv2d(ym_channel, num_channels_down, filter_size_down, padding ='same',padding_mode='reflect'),
            nn.BatchNorm2d(num_channels_down),
            # nn.LeakyReLU(0.2))
            nn.Softplus())

        self.enc = nn.Sequential(
            nn.Conv2d(num_channels_down, num_channels_down, filter_size_down,padding='same', padding_mode='reflect'),
            nn.BatchNorm2d(num_channels_down),
            # nn.LeakyReLU(0.2))
            nn.Softplus())

        self.skip = nn.Sequential(
            nn.Conv2d(num_channels_down, num_channels_skip, filter_skip_size, padding ='same', padding_mode='reflect'),
            nn.BatchNorm2d(num_channels_skip),
            # nn.LeakyReLU(0.2))
            nn.Softplus())

        self.dc = nn.Sequential(
            nn.Conv2d((num_channels_skip + num_channels_up), num_channels_up, filter_size_up,padding='same',padding_mode='reflect'),
            nn.BatchNorm2d(num_channels_up),
            # nn.LeakyReLU(0.2))
            nn.Softplus())
        self.out_layer = nn.Sequential(
            nn.Conv2d(num_channels_up, yh_channel, 1, padding_mode='reflect'),
            nn.Sigmoid())
        self.conv_hs = nn.Sequential(
            nn.Conv2d(yh_channel,num_channels_down,filter_size_down, padding = 'same',padding_mode = 'reflect'))
            # nn.BatchNorm2d(num_channels_down),
            # nn.Softplus())
        self.conv_bn = nn.Sequential(
            nn.Conv2d(num_channels_down,num_channels_down,filter_size_down, padding = 'same',padding_mode = 'reflect'),
            # nn.BatchNorm2d(num_channels_down),
            # # nn.LeakyReLU(0.2))
            nn.Softplus())
        self.ym_channels= ym_channel
    def forward(self, inputs):
        ym = inputs[:,:self.ym_channels,:,:]
        yh = inputs[:,self.ym_channels:,:,:]

        ym_en0 = self.guide_ms(ym)
        ym_en1 = self.enc(ym_en0)
        ym_en2 = self.enc(ym_en1)
        ym_en3 = self.enc(ym_en2)
        ym_en4 = self.enc(ym_en3)

        ym_dc0 = self.enc(ym_en4)
        ym_dc1 = self.enc(ym_dc0)
        ym_dc2 = self.dc(torch.cat((self.skip(ym_en4), ym_dc1), dim=1))
        ym_dc3 = self.dc(torch.cat((self.skip(ym_en3), ym_dc2), dim=1))
        ym_dc4 = self.dc(torch.cat((self.skip(ym_en2), ym_dc3), dim=1))
        ym_dc5 = self.dc(torch.cat((self.skip(ym_en1), ym_dc4), dim=1))
        ym_dc6 = self.dc(torch.cat((self.skip(ym_en0), ym_dc5), dim=1))

        # yh_0 = self.URU1(ym,yh) #
        # yh_0 = self.up_bic(yh)
        # yh_0 = self.up_trans(yh,output_size=(yh.shape[2]*4,yh.shape[3]*4))
        # yh_1 = self.FRU(ym_en0, self.conv_hs(yh_0))
        # yh_2 = self.FRU(ym_en1, yh_1)
        # yh_3 = self.FRU(ym_en2, yh_2)
        # yh_4 = self.FRU(ym_en3, yh_3)
        # yh_5 = self.FRU(ym_en4, yh_4)
        #
        yh_6 = self.FRU(self.conv_hs(yh), ym_dc0)
        yh_7 = self.FRU(self.conv_bn(yh_6), ym_dc1)
        yh_8 = self.FRU(self.conv_bn(yh_7), ym_dc2)
        yh_9 = self.FRU(self.conv_bn(yh_8), ym_dc3)
        yh_10 = self.FRU(self.conv_bn(yh_9), ym_dc4)
        yh_11 = self.FRU(self.conv_bn(yh_10), ym_dc5)
        yh_12 = self.FRU(self.conv_bn(yh_11), ym_dc6)

        # yh_6 = self.FRU(self.conv_hs(yh_0), ym_dc0)
        # yh_7 = self.FRU(yh_6, ym_dc1)
        # yh_8 = self.FRU(yh_7, ym_dc2)
        # yh_9 = self.FRU(yh_8, ym_dc3)
        # yh_10 = self.FRU(yh_9, ym_dc4)
        # yh_11 = self.FRU(yh_10, ym_dc5)
        # yh_12 = self.FRU(yh_11, ym_dc6)

        out = self.out_layer(yh_12)

        return out

class hs_net2(nn.Module):
    def __init__(self, ym_channel=1, yh_channel=93, num_channels_down=128, num_channels_up=128, num_channels_skip=64,
                 filter_size_down=3, filter_size_up=3, filter_skip_size=1):
        super(hs_net2,self).__init__()
        self.guide_ms = nn.Sequential(
            nn.Conv2d(ym_channel, num_channels_down, filter_size_down, padding ='same',padding_mode='reflect'),
            nn.BatchNorm2d(num_channels_down),
            # nn.LeakyReLU(0.2))
            nn.Softplus())

        self.enc = nn.Sequential(
            nn.Conv2d(num_channels_down, num_channels_down, filter_size_down,padding='same', padding_mode='reflect'),
            nn.BatchNorm2d(num_channels_down),
            # nn.LeakyReLU(0.2))
            nn.Softplus())

        self.skip = nn.Sequential(
            nn.Conv2d(num_channels_down, num_channels_skip, filter_skip_size, padding ='same', padding_mode='reflect'),
            nn.BatchNorm2d(num_channels_skip),
            # nn.LeakyReLU(0.2))
            nn.Softplus())

        self.dc = nn.Sequential(
            nn.Conv2d((num_channels_skip + num_channels_up), num_channels_up, filter_size_up,padding='same',padding_mode='reflect'),
            nn.BatchNorm2d(num_channels_up),
            # nn.LeakyReLU(0.2))
            nn.Softplus())
        self.out_layer = nn.Sequential(
            nn.Conv2d(num_channels_up, yh_channel, 1, padding_mode='reflect'),
            nn.Sigmoid())
        self.conv_hs = nn.Sequential(
            nn.Conv2d(yh_channel,num_channels_down,filter_size_down, padding = 'same',padding_mode = 'reflect'))
            # nn.BatchNorm2d(num_channels_down),
            # nn.Softplus())
        self.conv_bn = nn.Sequential(
            nn.Conv2d(num_channels_down,num_channels_down,filter_size_down, padding = 'same',padding_mode = 'reflect'),
            # nn.BatchNorm2d(num_channels_down),
            # # nn.LeakyReLU(0.2))
            nn.Softplus())
    def forward(self, inputs):
        ym = inputs[:,:4:,:,:]
        yh = inputs[:,4:,:,:]

        ym_en0 = self.guide_ms(ym)
        ym_en1 = self.enc(ym_en0)
        ym_en2 = self.enc(ym_en1)
        ym_en3 = self.enc(ym_en2)
        ym_en4 = self.enc(ym_en3)

        ym_dc0 = self.enc(ym_en4)
        ym_dc1 = self.enc(ym_dc0)
        ym_dc2 = self.dc(torch.cat((self.skip(ym_en4), ym_dc1), dim=1))
        ym_dc3 = self.dc(torch.cat((self.skip(ym_en3), ym_dc2), dim=1))
        ym_dc4 = self.dc(torch.cat((self.skip(ym_en2), ym_dc3), dim=1))
        ym_dc5 = self.dc(torch.cat((self.skip(ym_en1), ym_dc4), dim=1))
        ym_dc6 = self.dc(torch.cat((self.skip(ym_en0), ym_dc5), dim=1))

        # yh_0 = self.URU1(ym,yh) #
        yh_0 = self.up_bic(yh)
        # yh_0 = self.up_trans(yh,output_size=(yh.shape[2]*4,yh.shape[3]*4))
        # yh_1 = self.FRU(ym_en0, self.conv_hs(yh_0))
        # yh_2 = self.FRU(ym_en1, yh_1)
        # yh_3 = self.FRU(ym_en2, yh_2)
        # yh_4 = self.FRU(ym_en3, yh_3)
        # yh_5 = self.FRU(ym_en4, yh_4)
        #
        yh_6 = self.FRU(self.conv_hs(yh_0), ym_dc0)
        yh_7 = self.FRU(self.conv_bn(yh_6), ym_dc1)
        yh_8 = self.FRU(self.conv_bn(yh_7), ym_dc2)
        yh_9 = self.FRU(self.conv_bn(yh_8), ym_dc3)
        yh_10 = self.FRU(self.conv_bn(yh_9), ym_dc4)
        yh_11 = self.FRU(self.conv_bn(yh_10), ym_dc5)
        yh_12 = self.FRU(self.conv_bn(yh_11), ym_dc6)

        # yh_6 = self.FRU(self.conv_hs(yh_0), ym_dc0)
        # yh_7 = self.FRU(yh_6, ym_dc1)
        # yh_8 = self.FRU(yh_7, ym_dc2)
        # yh_9 = self.FRU(yh_8, ym_dc3)
        # yh_10 = self.FRU(yh_9, ym_dc4)
        # yh_11 = self.FRU(yh_10, ym_dc5)
        # yh_12 = self.FRU(yh_11, ym_dc6)

        out = self.out_layer(yh_12)

        return out