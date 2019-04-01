import pdb
import os
import numpy as np
import matplotlib.pyplot as plt
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax, create_pairwise_bilateral
import PIL.Image as Image
import multiprocessing
from evaluate import fm_and_mae
from skimage.segmentation import slic
from tqdm import tqdm
import cv2
from myfunc import make_graph
from scipy.sparse import coo_matrix, dia_matrix, eye
from scipy.sparse.linalg import inv, spsolve
from functools import reduce

# 7202, 7299 -> 7616
# 

theta = 10.0
alpha = 0.99

sal_set = 'ECSSD'
img_root = '../data/datasets/saliency_Dataset/%s/images'%sal_set
prob_root1 = '../ROTS2files/cap-init'
prob_root2 = '../ROTS2files/cls-init'
output_root = '../ROTS2files/init-sp-mr'

if not os.path.exists(output_root):
    os.mkdir(output_root)

files = os.listdir(img_root)


def mr_func(imgs, probs1, probs2):
    _, hh, ww, _ = imgs.shape
    msks = []
    for i, img in enumerate(imgs):
        prob1 = probs1[i]
        prob2 = probs2[i]

        # superpixel
        img_lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float) / 255.0
        sp_label = slic(img_lab, n_segments=200, compactness=20)
        # in case of empty superpixels
        sp_onehot = np.arange(sp_label.max() + 1) == sp_label[..., None]
        sp_onehot = sp_onehot[:, :, sp_onehot.sum(0).sum(0) > 0]
        rs, cs, num = np.where(sp_onehot)
        for i, n in enumerate(num):
            sp_label[rs[i], cs[i]] = n
        sp_num = sp_label.max() + 1
        sp_prob1 = []
        sp_prob2 = []
        sp_img = []
        for i in range(sp_num):
            sp_prob1.append(prob1[sp_label == i].mean())
            sp_prob2.append(prob2[sp_label == i].mean())
            # superpixel vector holds Lab value
            sp_img.append(img_lab[sp_label == i, :].mean(0, keepdims=False))
        sp_img = np.array(sp_img)
        sp_prob1 = np.array(sp_prob1)
        th1 = sp_prob1.mean()
        sp_prob2 = np.array(sp_prob2)
        th2 = sp_prob2.mean()
        seed = np.ones(sp_num)
        seed[sp_prob1<th1] = 0
        seed[sp_prob2<th2] = 0
        # affinity matrix
        edges = make_graph(sp_label)

        weight = np.sqrt(np.sum((sp_img[edges[:, 0]] - sp_img[edges[:, 1]]) ** 2, 1))
        weight = (weight - np.min(weight, axis=0, keepdims=True)) \
                 / (np.max(weight, axis=0, keepdims=True) - np.min(weight, axis=0, keepdims=True))
        weight = np.exp(-weight * theta)

        W = coo_matrix((
            np.concatenate((weight, weight)),
            (
                np.concatenate((edges[:, 0], edges[:, 1]), 0),
                np.concatenate((edges[:, 1], edges[:, 0]), 0)
            )))
        dd = W.sum(0)
        D = dia_matrix((dd, 0), (sp_num, sp_num)).tocsc()

        optAff = spsolve(D - alpha * W, eye(sp_num).tocsc())
        optAff -= dia_matrix((optAff.diagonal(), 0), (sp_num, sp_num))

        """stage 2"""
        fsal = optAff.dot(seed)
        fsal = (fsal - fsal.min()) / (fsal.max() - fsal.min())
        th = fsal.mean()
        fsal[fsal>th] = 1
        fsal[fsal<=th] = 0
        msk = np.zeros((hh, ww))
        for i in range(sp_num):
            msk[sp_label==i] = fsal[i]
        msks += [msk]
    msks = np.stack(msks, 0)
    return msks


def thisfunc(img_name):
    img = Image.open(os.path.join(img_root, img_name[:-4]+'.jpg')).convert('RGB')
    ww, hh = img.size
    img = np.array(img, dtype=np.uint8)
    probs1 = Image.open(os.path.join(prob_root1, img_name[:-4]+'.png'))
    probs1 = probs1.resize((ww, hh))
    probs1 = np.array(probs1)
    probs1 = probs1.astype(np.float)/255.0

    probs2 = Image.open(os.path.join(prob_root2, img_name[:-4]+'.png'))
    probs2 = probs2.resize((ww, hh))
    probs2 = np.array(probs2)
    probs2 = probs2.astype(np.float)/255.0

    # superpixel
    img_lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float) / 255.0
    sp_label = slic(img_lab, n_segments=200, compactness=20)
    # in case of empty superpixels
    sp_onehot = np.arange(sp_label.max() + 1) == sp_label[..., None]
    sp_onehot = sp_onehot[:, :, sp_onehot.sum(0).sum(0) > 0]
    rs, cs, num = np.where(sp_onehot)
    for i, n in enumerate(num):
        sp_label[rs[i], cs[i]] = n
    sp_num = sp_label.max() + 1
    sp_prob1 = []
    sp_prob2 = []
    sp_img = []
    for i in range(sp_num):
        sp_prob1.append(probs1[sp_label == i].mean())
        sp_prob2.append(probs2[sp_label == i].mean())
        # superpixel vector holds Lab value
        sp_img.append(img_lab[sp_label == i, :].mean(0, keepdims=False))
    sp_img = np.array(sp_img)
    sp_prob1 = np.array(sp_prob1)
    th1 = sp_prob1.mean()
    sp_prob2 = np.array(sp_prob2)
    th2 = sp_prob2.mean()
    seed = np.ones(sp_num)
    seed[sp_prob1<th1] = 0
    seed[sp_prob2<th2] = 0
    # bg1 = np.zeros(sp_num)
    # bg2 = np.zeros(sp_num)
    # bg1[sp_prob1<th1] = 1
    # bg2[sp_prob2<th2] = 1

    # affinity matrix
    edges = make_graph(sp_label)
    # edges = np.concatenate((np.stack((np.arange(sp_num), np.arange(sp_num)), 1), edges), 0)

    weight = np.sqrt(np.sum((sp_img[edges[:, 0]] - sp_img[edges[:, 1]]) ** 2, 1))
    weight = (weight - np.min(weight, axis=0, keepdims=True)) \
             / (np.max(weight, axis=0, keepdims=True) - np.min(weight, axis=0, keepdims=True))
    weight = np.exp(-weight * theta)

    W = coo_matrix((
        np.concatenate((weight, weight)),
        (
            np.concatenate((edges[:, 0], edges[:, 1]), 0),
            np.concatenate((edges[:, 1], edges[:, 0]), 0)
        )))
    dd = W.sum(0)
    D = dia_matrix((dd, 0), (sp_num, sp_num)).tocsc()

    optAff = spsolve(D - alpha * W, eye(sp_num).tocsc())
    optAff -= dia_matrix((optAff.diagonal(), 0), (sp_num, sp_num))

    # """stage 1"""
    # bds = [bg1, bg2]
    # bsal = []
    # for bd in bds:
    #     seed = np.zeros(sp_num)
    #     seed[bd] = 1
    #     _bsal = optAff.dot(seed)
    #     _bsal = (_bsal - _bsal.min()) / (_bsal.max() - _bsal.min())
    #     bsal.append(1 - _bsal)
    # bsal = reduce(lambda x, y: x * y, bsal)
    # bsal = (bsal - bsal.min()) / (bsal.max() - bsal.min())

    """stage 2"""
    fsal = optAff.dot(seed)
    fsal = (fsal - fsal.min()) / (fsal.max() - fsal.min())

    msk = np.zeros((hh, ww))
    for i in range(sp_num):
        msk[sp_label==i] = fsal[i]
    msk = (msk*255).astype(np.uint8)
    msk = Image.fromarray(msk)
    msk.save(os.path.join(output_root, img_name[:-4]+'.png'), 'png')


if __name__ == '__main__':
    # for file in tqdm(files):
    #     thisfunc(file)

    print('start crf')
    pool = multiprocessing.Pool(processes=8)
    pool.map(thisfunc, files)
    pool.close()
    pool.join()
    print('done')
    fm, mae, _, _ = fm_and_mae(output_root, '../data/datasets/saliency_Dataset/%s/masks'%sal_set)
    print(fm)
    print(mae)
