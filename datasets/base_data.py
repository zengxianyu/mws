import numpy as np
import PIL.Image as Image
from torch.utils import data
import pdb
import random


def rotated_rect_with_max_area(w, h, angle):
    """
    Given a rectangle of size wxh that has been rotated by 'angle' (in
    radians), computes the width and height of the largest possible
    axis-aligned rectangle (maximal area) within the rotated rectangle.
    """
    if w <= 0 or h <= 0:
        return 0, 0

    width_is_longer = w >= h
    side_long, side_short = (w, h) if width_is_longer else (h, w)

    # since the solutions for angle, -angle and 180-angle are all the same,
    # if suffices to look at the first quadrant and the absolute values of sin,cos:
    sin_a, cos_a = abs(np.sin(angle)), abs(np.cos(angle))
    if side_short <= 2. * sin_a * cos_a * side_long or abs(sin_a - cos_a) < 1e-10:
        # half constrained case: two crop corners touch the longer side,
        #   the other two corners are on the mid-line parallel to the longer line
        x = 0.5 * side_short
        wr, hr = (x / sin_a, x / cos_a) if width_is_longer else (x / cos_a, x / sin_a)
    else:
        # fully constrained case: crop touches all 4 sides
        cos_2a = cos_a * cos_a - sin_a * sin_a
        wr, hr = (w * cos_a - h * sin_a) / cos_2a, (h * cos_a - w * sin_a) / cos_2a

    return wr, hr


class BaseData(data.Dataset):
    def __init__(self, size=256, crop=None, rotate=None, flip=False,
                 mean=None, std=None):
        super(BaseData, self).__init__()
        self.mean, self.std = mean, std
        self.flip = flip
        self.rotate = rotate
        self.crop = crop
        if isinstance(size, tuple):
            self.size = size
        elif isinstance(size, int):
            self.size = (size, size)
        else:
            raise NotImplementedError

    def random_crop(self, *images):
        images = list(images)
        sz = [img.size for img in images]
        sz = set(sz)
        assert(len(sz)==1)
        w, h = sz.pop()
        th, tw = int(self.crop*h), int(self.crop*w)
        if w == tw and h == th:
            return 0, 0, h, w
        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        results = [img.crop((j, i, j + tw, i + th)) for img in images]
        return tuple(results)

    def random_flip(self, *images):
        if self.flip and random.randint(0, 1):
            images = list(images)
            results = [img.transpose(Image.FLIP_LEFT_RIGHT) for img in images]
            return tuple(results)
        else:
            return images

    def random_rotate(self, *images):
        images = list(images)
        sz = [img.size for img in images]
        sz = set(sz)
        assert(len(sz)==1)
        w, h = sz.pop()
        degree = random.randint(-1*self.rotate, self.rotate)
        images_r = [img.rotate(degree, expand=1) for img in images]
        w_b, h_b = images_r[0].size
        w_r, h_r = rotated_rect_with_max_area(w, h, np.radians(degree))
        ws = (w_b - w_r) / 2
        ws = max(ws, 0)
        hs = (h_b - h_r) / 2
        hs = max(hs, 0)
        we = ws + w_r
        he = hs + h_r
        we = min(we, w_b)
        he = min(he, h_b)
        results = [img.crop((ws, hs, we, he)) for img in images_r]
        return tuple(results)

    def __getitem__(self, item):
        pass

    def __len__(self):
        pass
