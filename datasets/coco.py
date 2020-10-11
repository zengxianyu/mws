import pdb
import torch
import os
import numpy as np
import nltk
from PIL import Image
from pycocotools.coco import COCO
from .base_data import BaseData


class CocoCaption(BaseData):
    """COCO Custom Dataset compatible with torch.utils.data.DataLoader."""
    def __init__(self, img_dir, gt_dir, vocab, size=256, training=True,
                 crop=None, rotate=None, flip=False, mean=None, std=None):
        """Set the path for images, captions and vocabulary wrapper.
        
        Args:
            root: image directory.
            json: coco annotation file path.
            vocab: vocabulary wrapper.
            transform: image transformer.
        """
        super(CocoCaption, self).__init__(size=size, crop=crop, rotate=rotate, flip=flip, mean=mean, std=std)
        self.root =img_dir
        self.coco = COCO(gt_dir)
        self.ids = list(self.coco.anns.keys())
        self.vocab = vocab

    def __getitem__(self, index):
        """Returns one data pair (image and caption)."""
        coco = self.coco
        vocab = self.vocab
        ann_id = self.ids[index]
        caption = coco.anns[ann_id]['caption']
        img_id = coco.anns[ann_id]['image_id']
        path = coco.loadImgs(img_id)[0]['file_name']

        img = Image.open(os.path.join(self.root, path)).convert('RGB')
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

        # Convert caption (string) to word ids.
        caption = caption.replace("\t", " ")
        tokens = nltk.tokenize.word_tokenize(str(caption).lower())
        caption = []
        caption.append(vocab('<start>'))
        caption.extend([vocab(token) for token in tokens])
        caption.append(vocab('<end>'))
        target = torch.Tensor(caption)
        return img, target

    def __len__(self):
        return len(self.ids)


def caption_collate_fn(data):
    """Creates mini-batch tensors from the list of tuples (image, caption).
    
    We should build custom collate_fn rather than using default collate_fn, 
    because merging caption (including padding) is not supported in default.

    Args:
        data: list of tuple (image, caption). 
            - image: torch tensor of shape (3, 256, 256).
            - caption: torch tensor of shape (?); variable length.

    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    # Sort a data list by caption length (descending order).
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions = zip(*data)

    # Merge images (from tuple of 3D tensor to 4D tensor).
    images = torch.stack(images, 0)

    # Merge captions (from tuple of 1D tensor to 2D tensor).
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]        
    return images, targets, lengths

