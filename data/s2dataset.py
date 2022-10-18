from utils.common_utils import *
from  utils.s2_utils import *
import torch
import numpy as np
import scipy.io as sio
import torch.utils.data as data

class S2Dataset(data.Dataset):
    def __init__(self, file_path='S2/australia.mat'):
        super(S2Dataset, self).__init__()
        self.file_path = file_path

        y10 = sio.loadmat(self.file_path)['y10'].astype(np.float32)
        y20 = sio.loadmat(self.file_path)['y20'].astype(np.float32)
        Y10 = hwc2chw(torch.from_numpy(y10))[None,:]
        Y20 = hwc2chw(torch.from_numpy(y20))[None,:]
        psf10 =get_s2psf(band='band10')
        psf20=get_s2psf(band='band20')

        self.y10 = compute_Hx(Y10,psf10,2)
        self.y20 = compute_Hx(Y20,psf20,2)
        self.ref = Y20
        # for i in range()



    # def generate_LrHSI(self, img, scale_factor):
    #     img_lr = self.downsamplePSF(img, sigma=self.args.sigma, stride=scale_factor)
    #     return img_lr
    #
    # def generate_HrMSI(self, img, sp_matrix):
    #     (h, w, c) = img.shape
    #     self.msi_channels = sp_matrix.shape[1]
    #     if sp_matrix.shape[0] == c:
    #         img_msi = np.dot(img.reshape(w*h,c), sp_matrix).reshape(h,w,sp_matrix.shape[1])
    #     else:
    #         raise Exception("The shape of sp matrix doesnot match the image")
    #     return img_msi

    def __getitem__(self, index):
        # img_patch = self.img_patch_list[index]
        # img_lr = self.img_lr_list[index]
        # img_msi = self.img_msi_list[index]
        # img_name = os.path.basename(self.imgpath_list[index]).split('.')[0]
        y10 = self.y10[index]
        y20 = self.y20[index]
        ref = self.ref[index]
        return {"y10":y10,
                'y20':y20,
                'ref': ref}

    def __len__(self):
        return len(self.y10)

dataset = S2Dataset()
first_sample = dataset[0]
y10=first_sample['y10']
y20=first_sample['y20']
ref = first_sample['ref']
print(dataset)
# print(y10.shape)
# print(y20.shape)
# print(ref.shape)
# fig1=plt.figure(figsize=(10,10))
# plt.imshow(y20[1,:,:],cmap='gray')
# plt.show()
# fig2=plt.figure(figsize=(10,10))
# plt.imshow(ref[1,:,:],cmap='gray')
# plt.show()