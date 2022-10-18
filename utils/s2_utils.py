import torch
import numpy as np
from utils.common_utils import *
import matplotlib.pyplot as plt
dtype = torch.cuda.FloatTensor

def get_s2psf(size =15, band='band20'):
    '''Get PSF of S2 bands, 20 m or 60 m'''
    d = np.array([6, 1, 1, 1, 2, 2, 2, 1, 2, 6, 2, 2])
    mtf = np.array([.32, .26, .28, .24, .38, .34, .34, .26, .33, .26, .22, .23])
    sdf = d * np.sqrt(-2 * np.log(mtf) / np.pi ** 2)
    # print(sdf[d==6])
    if band=='band20':
        sdf20 = sdf[d==2]
        psf20 = torch.zeros(len(sdf20),size,size)
        for i in range(len(sdf20)):
            psf20[i,:,:] = gaussian_filter(size,sdf20[i])
        return  psf20
    elif band=='band60':
        sdf60 = sdf[d == 6]
        psf60 = torch.zeros(len(sdf60), size, size)
        for i in range(len(sdf60)):
            psf60[i, :, :] = gaussian_filter(size, sdf60[i])
        return psf60
    elif band=='band10':
        sdf10 = sdf[d == 1]
        psf10 = torch.zeros(len(sdf10), size, size)
        for i in range(len(sdf10)):
            psf10[i, :, :] = gaussian_filter(size, sdf10[i])
        return psf10
    else:
        AssertionError

# def simulate_S2_data(Xm_im):
#     # Input:  Xm_im which is nc x nl x 12 tensor
#     # Output: Yim is a 12 x 1 cell array containing downsampled and
#     #         filtered bands of Xm_im
#     # convolution  operators (Gaussian convolution filters), taken from ref [5]
#     def create_conv_kernel(sdf, nl, nc, dx=12, dy=12):
#         # Create conv_kernel of size (dx,dy) with psf=sdf in fft domain for an image of size (nl,nc)
#         middlel = nl // 2
#         middlec = nc // 2
#         d = np.array([6, 1, 1, 1, 2, 2, 2, 1, 2, 6, 2, 2])
#         N = np.floor((dx + dy) / 2)
#         L = len(d)
#         B = torch.zeros([L, nl, nc])
#         for i in range(L):
#             if d[i] == 1:
#                 B[i, 0, 0] = 1
#                 FBM = torch.fft.fft2(B)
#             else:
#                 h = gaussian_filter(N=N, sigma=sdf[i])
#                 #             B[i, int(nl%2+1+(nl - dx - d[i])//2):int(nl%2+1+(nl + dx - d[i])//2), int(nc%2+1+(nc - dy - d[i])//2):int(nc%2+1+(nc + dy - d[i])//2)] = h
#                 B[i, int(nl % 2 + (nl - dx) // 2):int(nl % 2 + (nl + dx) // 2),
#                 int(nc % 2 + (nc - dy) // 2):int(nc % 2 + (nc + dy) // 2)] = h
#
#                 B[i, :, :] = torch.fft.fftshift(B[i, :, :])
#                 #             B[i,:,:] = np.divide(B[i,:,:],np.sum(B[i,:,:]))
#                 FBM = torch.fft.fft2(B)
#
#         return FBM
#
#     d = np.array([6, 1, 1, 1, 2, 2, 2, 1, 2, 6, 2, 2])
#     mtf = np.array([.32, .26, .28, .24, .38, .34, .34, .26, .33, .26, .22, .23])
#     dx = 12
#     dy = 12
#     [nl, nc, L] = Xm_im.shape
#     sdf = d * np.sqrt(-2 * np.log(mtf) / np.pi ** 2)
#     sdf[d == 1] = 0
#     X = Xm_im.permute(2, 0, 1)
#
#     # The following command creates the filter
#     FBM = create_conv_kernel(sdf, nl, nc, dx, dy)
#     # The following command performs the filtering
#     Xf = torch.real(torch.fft.ifft2(torch.fft.fft2(X) * FBM))
#     Xf=Xf.numpy()
#     # The following commands perform the downsampling
#     Yim = [Xf[idx, ::ratio, ::ratio] for idx, ratio in enumerate(d)]
#     return Yim

def sentinel2RGB(Yin, thresholdRGB):
    # Function create RGB image from Sentinel image
    # Y - a sentinel image with B2,B3 and B4
    # thresholdRGB =1, then the tail
    # and head of RGB values are removed.
    # This increases contrast

    [ydim, xdim, zdim] = np.shape(Yin)
    Y = np.transpose(np.reshape(Yin, (ydim * xdim, zdim)))
    Y = Y / np.max(Y)
    # T matrix from "Natural color representation of Sentinell-2 data " paper
    T = np.array([[0.180, 0.358, 0.412],
                  [0.072, 0.715, 0.213],
                  [0.950, 0.119, 0.019]])

    XYZ = np.matmul(T, Y)
    # % Convert to RGB
    M = np.array([[3.2404542, -1.5371385, -0.4985314],
                  [-0.9692660, 1.8760108, 0.0415560],
                  [0.0556434, -0.2040259, 1.0572252]])
    sRGB = np.matmul(M, XYZ)
    # % Correct gamma
    gamma_map = (sRGB > 0.0031308)
    sRGB[gamma_map] = 1.055 * np.power(sRGB[gamma_map], (1. / 2.4)) - 0.055
    sRGB[~gamma_map] = 12.92 * sRGB[~gamma_map]
    sRGB[sRGB > 1] = 1;
    sRGB[sRGB < 0] = 0;
    if (thresholdRGB > 0):
        thres = 0.01;
        for idx in range(3):
            y = sRGB[idx, :]
            [a, b] = np.histogram(y, bins=100)
            a = np.cumsum(a) / np.sum(a)
            th = b[0]
            i = np.argwhere(a < thres)
            if len(i) > 0:
                th = b[i[-1]]
            y = np.maximum(0, y - th)
            [a, b] = np.histogram(y, bins=100)
            a = np.cumsum(a) / np.sum(a)
            i = np.argwhere(a > 1 - thres)
            th = b[i[0]]
            y[y > th] = th
            y = y / th
            sRGB[idx, :] = y

    RGB = np.zeros_like(Yin)
    RGB[:, :, 0] = np.reshape(sRGB[0, :], (ydim, xdim))
    RGB[:, :, 1] = np.reshape(sRGB[1, :], (ydim, xdim))
    RGB[:, :, 2] = np.reshape(sRGB[2, :], (ydim, xdim))
    return RGB
