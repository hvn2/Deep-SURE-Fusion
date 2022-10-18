import torch
import numpy as np
from utils.common_utils import *

def get_hs_psf(N=7,n_channels=93, gaussian=True,sigma=2.):
    '''Assume all HS bands have the same psf
    which is Gaussian with sigma=1.'''
    psf = torch.zeros(n_channels,N,N)
    if gaussian:
        for i in range(n_channels):
            psf[i,:,:]=gaussian_filter(N=N,sigma=sigma)
    else:
        for i in range(n_channels):
            psf[i,:,:]=get_filter()
    return psf

def get_sigma(sigma=1,eta=20,band=93):
    '''Generate \sigma as a bell-curve function
    sigma_i**2 = sigma**exp(-(i-B/2)**2/(2*eta**2))/sum(exp(-(i-B/2)**2/(2*eta**2))'''
    den = 0
    num =[]
    for i in range(1,band+1):
        den = den + np.exp(-(i-band/2)**2/(2*eta**2))
    for k in range(band):
        num.append(np.exp(-(k-band/2)**2/(2*eta**2)))
    return np.sqrt(sigma**2*(num/den))
def add_bandwise_noise(x,sigma,device='cuda'):
    '''Add noise with std. sigma to image x (NCHW)'''
    if isinstance(sigma, np.ndarray):
        sigma = torch.from_numpy(sigma).to(device)
    else:
        sigma=sigma.to(device)
    xnoise = torch.zeros_like(x)
    for i in range(len(sigma)):
        xnoise[i,:,:] = x[i,:,:]+torch.randn(size=(x.shape[1],x.shape[1]),device=device)*sigma[i]
    return xnoise

def reshape_fortran(x, shape):
    if len(x.shape) > 0:
        x = x.permute(*reversed(range(len(x.shape))))
    return x.reshape(*reversed(shape)).permute(*reversed(range(len(shape))))

def im2mat(X):
    """X(r,c,b)-->X(r*c,b)"""
    mat=reshape_fortran(X,(X.shape[0]*X.shape[1],X.shape[2]))
    return mat

def mat2im(X,r):
    """X(r*c,b)-->X(r*c,b)"""
    c=int(X.shape[0]/r)
    b=X.shape[1]
    return reshape_fortran(X,(r,c,b))
def hsi2msi(X,R):
    """Convert HSI X to MSI M by multiplying with spectral response R: M=X*R
    X: (HxWxNh), R(NhxNm)"""
    [r,c,b]=X.size()
    x=im2mat(X)
    if R.shape[0]!=b:
        R=torch.transpose(R,0,1)
    xout = torch.mm(x,R)
    return mat2im(xout,r)

def SNRCal(Xref,X):
    '''Calculate signal to reconstructed error (SRE) between reference image and reconstructed image
    Input: Xref, X: reference and reconstructed images in shape [h,w,d]
    Output: aSRE average SRE in dB
            SRE_vec: SRE of each band'''
    Xref=Xref.reshape(Xref.shape[0]*Xref.shape[1]*Xref.shape[2])
    X=X.reshape(X.shape[0]*X.shape[1]*X.shape[2])
    return  10*np.log(np.sum(Xref**2)/np.sum((Xref-X)**2))/np.log(10)