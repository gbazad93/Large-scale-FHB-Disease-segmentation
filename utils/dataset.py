from os.path import splitext
from os import listdir
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset
import logging
from PIL import Image
import cv2

class BasicDataset(Dataset):
    def __init__(self, imgs_dir, masks_dir, scale=1, mask_suffix=''):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.scale = scale
        self.mask_suffix = mask_suffix
        
        
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'

        self.ids = [splitext(file)[0] for file in listdir(imgs_dir)
                    if not file.startswith('.')]
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, pil_img, scale):    
        
        img_nd = pil_img

        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)
        
        if img_nd.max() > 1:
            img_nd = img_nd / 255

        img_nd = np.array(img_nd)
        img_nd = np.einsum('ijk->kij', img_nd)
        return img_nd
    
    def mask_to_class(self, mask):
        mask2 = np.zeros((3, mask.shape[1],mask.shape[2]))
        mask2[0, :,:] = np.where(mask[:,:,1]+mask[:,:,2]>0, 0, 1)
        mask2[1, :,:] = mask[:,:,1]
        mask2[2, :,:] = mask[:,:,2]
        
       
        mask2 = mask2*255
        return mask2
    

    def __getitem__(self, i):
        idx = self.ids[i]
        mask_file = glob(self.masks_dir + idx + self.mask_suffix + '.*')
        img_file = glob(self.imgs_dir + idx + '.*')

        assert len(mask_file) == 1, \
            f'Either no mask or multiple masks found for the ID {idx}: {mask_file}'
        assert len(img_file) == 1, \
            f'Either no image or multiple images found for the ID {idx}: {img_file}'
        mask = cv2.imread(mask_file[0], cv2.COLOR_BGR2RGB)
        mask = cv2.resize(mask, (1024,1024), interpolation = cv2.INTER_AREA)
        # mask  = np.where(mask>0, 1, 0)
        mask2= mask[:,:,:]
        mask2[np.all(mask2 == (21, 21, 21), axis=-1)] = (0,255,0)
        mask2[np.all(mask2 == (25, 25, 25), axis=-1)] = (255,0,0)
        mask2=np.einsum('ijk->kij', mask2)
        mask2 = mask2 / 255
        mask2 = np.where(mask2 > .99, 1, 0)
        
        
        img = cv2.imread(img_file[0], cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (1024,1024), interpolation = cv2.INTER_AREA)
        

        assert img.size == mask.size, \
            f'Image and mask {idx} should be the same size, but are {img.size} and {mask.size}'

        img = self.preprocess(img, self.scale)
#         img=img[0:1,:,:]
        img=img*255
        
        mask = self.preprocess(mask, self.scale)
        
        
#         mask = self.mask_to_class(mask)
#         mask=mask[0,:,:]
       



        return {
           'image': torch.from_numpy(img).type(torch.FloatTensor),
            'mask': torch.from_numpy(mask2).type(torch.FloatTensor)
        }

