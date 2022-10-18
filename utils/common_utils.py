import torch
import numpy as np
from torch.fft import fft2 as fft2
from torch.fft import ifft2 as ifft2
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def hwc2chw(x):
    return(x.permute(2,0,1))
def chw2hwc(x):
    return(x.permute(1,2,0))
def get_filter():
    h = np.array([[1, 4, 6, 4, 1], [4, 16, 24, 16, 4], [6, 24, 36, 24, 6], [4, 16, 24, 16, 4], [1, 4, 6, 4, 1]])
    return torch.from_numpy(h/np.sum(h))
def gaussian_filter(N=15, sigma=2.0):
    n = (N - 1) / 2.0
    y, x = np.ogrid[-n:n + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2 * sigma ** 2))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return torch.from_numpy(h)

def sreCal(Xref,X):
    '''Calculate signal to reconstructed error (SRE) between reference image and reconstructed image
    Input: Xref, X: reference and reconstructed images in shape [h,w,d]
    Output: aSRE average SRE in dB
            SRE_vec: SRE of each band'''
    mSRE=0
    if len(Xref.shape)==3:
        Xref=Xref.reshape(Xref.shape[0]*Xref.shape[1],Xref.shape[2])
        X=X.reshape(X.shape[0]*X.shape[1],X.shape[2])
        SRE_vec=np.zeros((X.shape[1]))
        for i in range(X.shape[1]):
            SRE_vec[i]=10*np.log(np.sum(Xref[:,i]**2)/np.sum((Xref[:,i]-X[:,i])**2))/np.log(10)
        mSRE=np.mean(SRE_vec)
    else:
        pass
    return mSRE

def add_noise(x,SNRdb=40.):
    '''Add noise SNRdb to clean image x (CHW)'''
    x=x.to(device)
    SNR = 10**(SNRdb/10.)
    sigma = torch.sqrt(torch.mean(x**2)/SNR)
    sigma = sigma.to(device)
    xnoise = x + sigma*torch.randn(x.shape).to(device)
    return xnoise,sigma

def fill_noise(x, noise_type):
    """Fills tensor `x` with noise of type `noise_type`."""
    if noise_type == 'u':
        x.uniform_()
    elif noise_type == 'n':
        x.normal_()
    else:
        assert False
def get_noise(input_depth, method, spatial_size, noise_type='u', var=1. / 10):
    """Returns a pytorch.Tensor of size (1 x `input_depth` x `spatial_size[0]` x `spatial_size[1]`)
    initialized in a specific way.
    Args:
        input_depth: number of channels in the tensor
        method: `noise` for fillting tensor with noise; `meshgrid` for np.meshgrid
        spatial_size: spatial size of the tensor to initialize
        noise_type: 'u' for uniform; 'n' for normal
        var: a factor, a noise will be multiplicated by. Basically it is standard deviation scaler.
    """
    if isinstance(spatial_size, int):
        spatial_size = (spatial_size, spatial_size)

    if method == '2D':
        shape = [1, input_depth, spatial_size[0], spatial_size[1]]
    elif method == '3D':
        shape = [1, 1, input_depth, spatial_size[0], spatial_size[1]]
    else:
        assert False

    net_input = torch.zeros(shape)

    fill_noise(net_input, noise_type)
    net_input *= var

    return net_input

def expand_shift_psf(psf,c,m,n):
    '''pad psf to (mxn) with zeros, move the psf to the pad-center and shift--> reduce the boundary effect after convolution
    input: PSF (small size)
    output: PSF expand and shift to (c x m x n)'''
    midm=m//2
    midn=n//2
    if len(psf.shape)<3:
        midx=psf.shape[0]//2
        midy=psf.shape[1]//2
        y = torch.zeros(c, m, n)
    else:
        midx = psf.shape[1] // 2
        midy = psf.shape[2] // 2
        y = torch.zeros(psf.shape[0], m, n)
    y[:, midm - midx :midm + midx+1, midn - midy:midn + midy+1] = psf
    return torch.fft.fftshift(y,dim=[1,2])

def Ax(img,psf,ratio):
    '''Filtering and downsampling by A
    input: image (c x m x n); psf: PSF (); ratio: downsampling ratio'''
    c,m,n = img.shape
    x = img.to(device)
    h = expand_shift_psf(psf,c,m,n).to(device)
    img = torch.real(ifft2(fft2(x)*fft2(h))) #filtering
    imgd = img[:,::ratio,::ratio] #downsampling
    return imgd
def zeros_upsampling(img,ratio):
    """Upsample by inserting zeros between samples
    input: img (c x m x n), ratio: upsampling ratio"""
    c,m,n=img.shape
    x = img.to(device)
    y = torch.zeros(c,m*ratio,n*ratio).to(device)
    y[:,::ratio,::ratio]=x
    return y

def Atx(img,psf,ratio):
    '''upsampling by inserting zeros between samples and filtering
    input: img (c x m x n), ratio: upsampling ratio; psf: PSF()'''
    c, m, n = img.shape
    x = img.to(device)
    y = torch.zeros(c, m * ratio, n * ratio).to(device)
    y[:, ::ratio, ::ratio] = x # y=upsampling by inserting zeros
    h = expand_shift_psf(psf,y.shape[0],y.shape[1],y.shape[2]).to(device)
    img = torch.real(ifft2(fft2(y)*torch.conj(fft2(h))))
    return img

# def AAt(psf,ratio,c,m,n):
#     '''compute the filtering and downsampling operator in FFT domain
#     HHt=MB(MB)t=MBBtMt is equivalent to 0th polyphase of BBt (blurring)[cite]
#     Input: psf: PSF; ratio: downsampling factor, (c,m,n): filter size expanded to image
#     Return: filter in FFT domain
#     '''
#     h=expand_shift_psf(psf,c,m,n).to(device)
#     h0=torch.real(ifft2(abs(fft2(h))**2))
#     h0d = h0[:,::ratio,::ratio]
#     return fft2(h0d)

def AAtx(img,psf,ratio):
    '''Compute AAt applying to image x [(AAt)^-1x] which is equivalent to 0th polyphase of BBt (blurring)[cite]
    Input: img: image (c x m x n), ratio, downsampling factor, h: PSF, cond: condition number
    output: inverse filtered image (c x m*ratio, n*ratio)'''
    c,m,n = img.shape
    x = img.to(device)
    nom = fft2(x)
    h = expand_shift_psf(psf, c, m*ratio, n*ratio).to(device)
    h0 = torch.real(ifft2(abs(fft2(h)) ** 2)) #BBT
    h0d = h0[:, ::ratio, ::ratio] #0 th component of the polyphase decomposition BBT
    aat = fft2(h0d)
    img_out = torch.real(ifft2(nom*aat))
    return img_out

def invAAtx(img,psf,ratio,cond):
    '''Compute inverse AAt applying to image x [(AAt)^-1x] which is equivalent to inverse 0th polyphase of BBt (blurring)[cite]
    Input: img: image (c x m x n), ratio, downsampling factor, h: PSF, cond: condition number
    output: inverse filtered image (c x m*ratio, n*ratio)'''
    c,m,n = img.shape
    x = img.to(device)
    nom = fft2(x)
    h = expand_shift_psf(psf, c, m*ratio, n*ratio).to(device)
    h0 = torch.real(ifft2(abs(fft2(h)) ** 2)) #BBT
    h0d = h0[:, ::ratio, ::ratio] #0 th component of the polyphase decomposition BBT
    aat = fft2(h0d)
    dem = aat+cond
    img_out = torch.real(ifft2(nom/dem))
#     dem=torch.where(torch.abs(aat)>cond,1/torch.abs(aat),torch.tensor(0,dtype=torch.float32).to(device))
#     img_out = torch.real(ifft2(nom*dem))
    return img_out

def projx(x,psf,ratio,cond):
    '''projection operator P = H^T(HH^T)^-1H=H+H apply to x
    Input: x: image (c,m,n),psf: PSF, ratio: up/downsampling factor; cond: condition number'''
    a = Ax(x,psf,ratio)
    b = invAAtx(a,psf,ratio,cond)
    return Atx(b,psf,ratio)

def back_projx(x,psf,ratio,cond):
    a = invAAtx(x,psf,ratio,cond)
    return Atx(a,psf,ratio)