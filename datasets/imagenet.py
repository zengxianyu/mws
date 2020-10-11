import os
import numpy as np
import PIL.Image as Image
import torch
from .base_data import BaseData
import pdb


class ImageFolders(BaseData):
    def __init__(self, root, size=256, training=True,
                 crop=None, rotate=None, flip=False, mean=None, std=None):
        super(ImageFolders, self).__init__(size=size, crop=crop, rotate=rotate, flip=flip, mean=mean, std=std)
        self.root = root
        self.training = training
        folders = os.listdir(root)
        self.img_filenames = [os.path.join(root, os.path.join(f, n))
                              for f in folders for n in os.listdir(os.path.join(root, f))]

    def __len__(self):
        return len(self.img_filenames)

    def __getitem__(self, index):
        # load image
        img_file = self.img_filenames[index]
        img = Image.open(img_file).convert('RGB')
        if self.crop is not None:
            img, = self.random_crop(img)
        if self.rotate is not None:
            img, = self.random_rotate(img)
        if self.flip:
            img, = self.random_flip(img)
        img = img.resize(self.size)
        img = np.array(img, dtype=np.float64) / 255.0
        if len(img.shape) < 3:
            img = np.stack((img, img, img), 2)
        if img.shape[2] > 3:
            img = img[:, :, :3]
        if self.mean is not None:
            img -= self.mean
        if self.std is not None:
            img /= self.std
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).float()
        return img


class ImageFiles(BaseData):
    def __init__(self, root, size=256, training=True,
                 crop=None, rotate=None, flip=False, mean=None, std=None):
        super(ImageFiles, self).__init__(size=size, crop=crop, rotate=rotate, flip=flip, mean=mean, std=std)
        self.root = root
        names = os.listdir(root)
        self.img_filenames = list(map(lambda x: os.path.join(root, x), names))
        names = list(map(lambda x: '.'.join(x.split('.')[:-1]), names))
        self.names = names

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        # load image
        img_file = self.img_filenames[index]
        img = Image.open(img_file).convert('RGB')
        if self.crop is not None:
            img, = self.random_crop(img)
        if self.rotate is not None:
            img, = self.random_rotate(img)
        if self.flip:
            img, = self.random_flip(img)
        img = img.resize(self.size)
        img = np.array(img, dtype=np.float64) / 255.0
        if len(img.shape) < 3:
            img = np.stack((img, img, img), 2)
        if img.shape[2] > 3:
            img = img[:, :, :3]
        if self.mean is not None:
            img -= self.mean
        if self.std is not None:
            img /= self.std
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).float()
        return img


class ImageNetDetCls(BaseData):
    def __init__(self, devkit_dir, size=256, training=True,
                 crop=None, rotate=None, flip=False, mean=None, std=None):
        super(ImageNetDetCls, self).__init__(size=size, crop=crop, rotate=rotate, flip=flip, mean=mean, std=std)
        self.root = devkit_dir
        txts = os.listdir(os.path.join(devkit_dir, 'data', 'det_lists'))
        txts = filter(lambda x: x.startswith('train_pos') or x.startswith('train_part'), txts)
        file2lbl = {}
        for txt in txts:
            files = open(os.path.join(devkit_dir, 'data', 'det_lists', txt)).readlines()
            for f in files:
                f = f.strip('\n')+'.JPEG'
                if f in file2lbl:
                    file2lbl[f] += [int(txt.split('.')[0].split('_')[-1])]
                else:
                    file2lbl[f] = [int(txt.split('.')[0].split('_')[-1])]
        self.file2lbl = file2lbl.items()

    def __len__(self):
        return len(self.file2lbl)

    def __getitem__(self, index):
        # load image
        img_file, lbl = self.file2lbl[index]
        img = Image.open(os.path.join(self.root, 'images', img_file)).convert('RGB')

        if self.crop is not None:
            img, = self.random_crop(img)
        if self.rotate is not None:
            img, = self.random_rotate(img)
        if self.flip:
            img, = self.random_flip(img)
        img = img.resize(self.size)

        img = np.array(img, dtype=np.float64) / 255.0
        if len(img.shape) < 3:
            img = np.stack((img, img, img), 2)
        if img.shape[2] > 3:
            img = img[:, :, :3]
        if self.mean is not None:
            img -= self.mean
        if self.std is not None:
            img /= self.std
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).float()

        onehot = np.zeros(200)
        lbl = np.array(lbl)-1
        onehot[lbl] = 1
        onehot = torch.from_numpy(onehot).float()
        return img, onehot


if __name__ == "__main__":
    sb = ImageNetDetCls('../../data/datasets/ILSVRC2014_devkit')
    img, gt = sb.__getitem__(0)
    pdb.set_trace()
