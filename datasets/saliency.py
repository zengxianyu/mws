import os
import numpy as np
import PIL.Image as Image
import torch
import pdb
from .base_data import BaseData


class Folder(BaseData):
    def __init__(self, img_dir, gt_dir, img_format='jpg', gt_format='png', size=256, training=True,
                 crop=None, rotate=None, flip=False, mean=None, std=None):
        super(Folder, self).__init__(size=size, crop=crop, rotate=rotate, flip=flip, mean=mean, std=std)
        names1 = ['.'.join(name.split('.')[:-1]) for name in os.listdir(gt_dir)]
        names2 = ['.'.join(name.split('.')[:-1]) for name in os.listdir(img_dir)]
        names = list(set(names1)&set(names2))
        self.img_filenames = [os.path.join(img_dir, name+'.'+img_format) for name in names]
        self.gt_filenames = [os.path.join(gt_dir, name+'.'+gt_format) for name in names]
        self.names = names
        self.training = training

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        # load image
        img_file = self.img_filenames[index]
        img = Image.open(img_file)
        gt_file = self.gt_filenames[index]
        name = self.names[index]
        gt = Image.open(gt_file)
        WW, HH = gt.size
        img = img.resize((WW, HH))
        if self.crop is not None:
            img, gt = self.random_crop(img, gt)
        if self.rotate is not None:
            img, gt = self.random_rotate(img, gt)
        if self.flip:
            img, gt = self.random_flip(img, gt)
        img = img.resize(self.size)
        gt = gt.resize(self.size)

        img = np.array(img, dtype=np.float64) / 255.0
        gt = np.array(gt, dtype=np.uint8)
        gt[gt != 0] = 1
        if len(img.shape) < 3:
            img = np.stack((img, img, img), 2)
        if img.shape[2] > 3:
            img = img[:, :, :3]
        if len(gt.shape) > 2:
            gt = gt[:, :, 0]
        if self.mean is not None:
            img -= self.mean
        if self.std is not None:
            img /= self.std
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).float()
        gt = torch.from_numpy(gt).float()
        if self.training:
            return img, gt
        else:
            return img, name, WW, HH
