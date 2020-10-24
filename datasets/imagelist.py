import os
import numpy as np
import PIL.Image as Image
import torch
import pdb
from .base_data import BaseData


class ImageList(BaseData):
    def __init__(self, img_dir, img_list, img_format='jpg', size=256, training=True,
                 crop=None, rotate=None, flip=False, mean=None, std=None):
        super(ImageList, self).__init__(size=size, crop=crop, rotate=rotate, flip=flip, mean=mean, std=std)
        with open(img_list, "r") as f:
            names = f.readlines()
        filenames = ["{}/{}.{}".format(img_dir, n.strip("\n"), img_format) for n in names]
        self.filenames = filenames
        self.training = training

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        # load image
        img_file = self.filenames[index]
        img = Image.open(img_file).convert("RGB")
        WW, HH = img.size
        img = img.resize((WW, HH))
        if self.crop is not None:
            img, = self.random_crop(img)
        if self.rotate is not None:
            img, = self.random_rotate(img)
        if self.flip:
            img, = self.random_flip(img)
        img = img.resize(self.size)

        img = np.array(img, dtype=np.float64) / 255.0
        if self.mean is not None:
            img -= self.mean
        if self.std is not None:
            img /= self.std
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).float()
        return img
