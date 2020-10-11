import numpy as np
from skimage.segmentation import slic
import cv2
from scipy.sparse import coo_matrix, dia_matrix, eye
from scipy.sparse.linalg import spsolve
from PIL import Image
import os
import torch.nn.functional as F
from tqdm import tqdm
import multiprocessing
from functools import partial
import pdb
import torchvision
import torch
import sys
sys.path.append('../../../')
from evaluate_sal import fm_and_mae
from models.networks.densenet import densenet169

# def deep_mr():


def img_from_sp(sp_img, sp_label):
    """
    restore an image from superpixels
    :param sp_img: superpixel image
    :param sp_label: superpixel segmentaion label maps
    :return: rimg: restored image
    """
    assert len(sp_img)== sp_label.max()+1, "inconsistent superpixel number"
    rimg = np.zeros(sp_label.shape+(3,))
    for i in range(len(sp_img)):
        rimg[sp_label==i, :] = sp_img[i]
    return rimg.astype('uint8')


def lbmap_from_sp(sp_lbmap, sp_label):
    """
    restore a label map from superpixels
    :param sp_lbmap: superpixel-level label
    :param sp_label: superpixel segmentation label
    :return: r_lbmap: restored label map
    """
    assert len(sp_lbmap)== sp_label.max()+1, "inconsistent superpixel number"
    r_lbmap = np.zeros(sp_label.shape)
    for i in range(len(sp_lbmap)):
        r_lbmap[sp_label==i] = sp_lbmap[i]
    return r_lbmap


def adjacent_matrix(sp_label):
    """
    adjacent matrix of superpixels
    :param sp_label:
    :return: edges: edges of the undirected graph
    """
    # should be improve
    sp_label_l = np.zeros(sp_label.shape, dtype='int')
    sp_label_u = np.zeros(sp_label.shape, dtype='int')
    sp_label_u[:-1, :] = sp_label[1:, :]
    sp_label_l[:, :-1] = sp_label[:, 1:]

    dl = sp_label_l - sp_label
    dl[:, -1] = 0

    du = sp_label_u - sp_label
    du[-1, :] = 0

    node_out = np.concatenate((sp_label[dl != 0], sp_label[du != 0]))
    node_in = np.concatenate((sp_label_l[dl != 0], sp_label_u[du != 0]))
    edges = np.stack((node_out, node_in), 1)
    edges = np.concatenate((edges, edges[:, ::-1]), 0)
    edges = np.unique(edges, axis=0)
    return edges


def make_graph(sp_label):
    """
    add boundary connections and far connections to the graph
    :param sp_label: superpixel segmentation
    :return: edges: edges of the new undirected graph
    """
    edges = adjacent_matrix(sp_label)
    sp_num = sp_label.max()

    # add boundary connections
    top_bd = np.unique(sp_label[0, :])
    left_bd = np.unique(sp_label[:, 0])
    bottom_bd = np.unique(sp_label[-1, :])
    right_bd = np.unique(sp_label[:, -1])

    boundary = np.concatenate((top_bd, left_bd, bottom_bd, right_bd))
    boundary = np.unique(boundary)

    nb = len(boundary)
    node_out = boundary.repeat(nb)
    node_in = node_out.reshape((nb, nb)).T.ravel()

    bd_edges = np.stack((node_out, node_in), axis=1)
    bd_edges = bd_edges[bd_edges[:, 0] != bd_edges[:, 1]]

    edges = np.concatenate((edges, bd_edges), 0)
    edges = np.unique(edges, axis=0)

    # add far connections
    far_in = []
    far_out = []
    for i in range(sp_num):
        temp = edges[edges[:, 0] == i, 1]
        _far_in = []
        for t in temp:
            _far_in.append(edges[edges[:, 0] == t, 1])
        _far_in = np.concatenate(_far_in)
        _far_out = np.ones(_far_in.shape, dtype='int') * i
        far_in.append(_far_in)
        far_out.append(_far_out)

    far_in = np.concatenate(far_in)
    far_out = np.concatenate(far_out)
    far_edges = np.stack((far_out, far_in), 1)
    far_edges = np.unique(far_edges, axis=0)
    far_edges = far_edges[far_edges[:, 0] != far_edges[:, 1]]

    edges = np.concatenate((edges, far_edges), 0)
    edges = np.array(edges)
    return edges


def mr(imgs, list_probs, alpha=0.99, theta=10.0):
    _, hh, ww, _ = imgs.shape
    msks = []
    for i, img in enumerate(imgs):
        # superpixel
        list_prob = [probs[i] for probs in list_probs]
        img_lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float) / 255.0
        sp_label = slic(img_lab, n_segments=200, compactness=20)
        # in case of empty superpixels
        sp_onehot = np.arange(sp_label.max() + 1) == sp_label[..., None]
        sp_onehot = sp_onehot[:, :, sp_onehot.sum(0).sum(0) > 0]
        rs, cs, num = np.where(sp_onehot)
        for i, n in enumerate(num):
            sp_label[rs[i], cs[i]] = n
        sp_num = sp_label.max() + 1
        list_sp_probs = []
        sp_img = []
        for t in range(len(list_probs)):
            sp_probs = []
            for j in range(sp_num):
                sp_probs.append(list_prob[t][sp_label == j].mean())
                # superpixel vector holds Lab value
                sp_img.append(img_lab[sp_label == j, :].mean(0, keepdims=False))
            list_sp_probs.append(sp_probs)
        sp_img = np.array(sp_img)
        list_sp_probs = [np.array(sp_probs) for sp_probs in list_sp_probs]
        seed = np.ones(sp_num)
        for sp_probs in list_sp_probs:
            th = sp_probs.mean()
            seed[sp_probs<th] = 0
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


def deep_mr(imgs, feats, list_probs, alpha=0.99, theta=10.0):
    _, hh, ww, _ = imgs.shape
    msks = []
    for i, img in enumerate(imgs):
        # superpixel
        feat = feats[i]
        list_prob = [probs[i] for probs in list_probs]
        img_lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float) / 255.0
        sp_label = slic(img_lab, n_segments=200, compactness=20)
        # in case of empty superpixels
        sp_onehot = np.arange(sp_label.max() + 1) == sp_label[..., None]
        sp_onehot = sp_onehot[:, :, sp_onehot.sum(0).sum(0) > 0]
        rs, cs, num = np.where(sp_onehot)
        for i, n in enumerate(num):
            sp_label[rs[i], cs[i]] = n
        sp_num = sp_label.max() + 1
        list_sp_probs = []
        sp_feat = []
        for t in range(len(list_probs)):
            sp_probs = []
            for j in range(sp_num):
                sp_probs.append(list_prob[t][sp_label == j].mean())
                # superpixel vector holds Lab value
                sp_feat.append(feat[sp_label==j, :].mean(0, keepdims=False))
            list_sp_probs.append(sp_probs)
        sp_feat = np.array(sp_feat)
        list_sp_probs = [np.array(sp_probs) for sp_probs in list_sp_probs]
        seed = np.ones(sp_num)
        for sp_probs in list_sp_probs:
            th = sp_probs.mean()
            seed[sp_probs<th] = 0
        # affinity matrix
        edges = make_graph(sp_label)

        weight = np.sqrt(np.sum((sp_feat[edges[:, 0]] - sp_feat[edges[:, 1]]) ** 2, 1))
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


def proc_img(name, img_root, prob_root, output_root):
    img = Image.open(os.path.join(img_root, name+'.jpg')).convert('RGB')
    ww, hh = img.size
    img = np.array(img, dtype=np.uint8)
    prob = Image.open(os.path.join(prob_root, name+'.png'))
    prob = prob.resize((ww, hh))
    prob = np.array(prob)
    prob = prob.astype(np.float)/255.0
    msk = mr(img[None, ...], [[prob]])
    msk = msk[0]
    msk = Image.fromarray((msk*255).astype(np.uint8))
    msk.save('{}/{}.png'.format(output_root, name))
    return msk


def proc_feat(name, feats, img_root, prob_root, output_root):
    img = Image.open(os.path.join(img_root, name+'.jpg')).convert('RGB')
    ww, hh = img.size
    img = np.array(img, dtype=np.uint8)
    prob = Image.open(os.path.join(prob_root, name+'.png'))
    prob = prob.resize((ww, hh))
    prob = np.array(prob)
    prob = prob.astype(np.float)/255.0
    msk = deep_mr(img[None, ...], feats, [[prob]])
    msk = msk[0]
    msk = Image.fromarray((msk*255).astype(np.uint8))
    msk.save('{}/{}.png'.format(output_root, name))
    return msk


if __name__ == "__main__":
    img_root = '/home/zeng/data/datasets/saliency_Dataset/ECSSD/images'
    prob_root = '/home/zeng/mwsFiles/clscap/results'
    output_root = '/home/zeng/mwsFiles/clscap/mr_f23'
    if not os.path.exists(output_root):
        os.mkdir(output_root)
    img_names = os.listdir(img_root)
    names = ['.'.join(n.split('.')[:-1]) for n in img_names]
    print('start')
    # pool = multiprocessing.Pool(processes=8)
    # pool.map(partial(proc_img, img_root=img_root, prob_root=prob_root, output_root=output_root, mr_func=mr), names)
    # pool.close()
    # pool.join()
    net = densenet169(pretrained=True)
    net.cuda()
    net.feats = []
    def hook(module, input, output):
        net.feats += [output]
    net.features.transition3[-2].register_forward_hook(hook)
    net.features.transition2[-2].register_forward_hook(hook)
    net.features.transition1[-2].register_forward_hook(hook)
    net.features.block0[-2].register_forward_hook(hook)
    net.features.transition3[-1].kernel_size = 1
    net.features.transition3[-1].stride = 1

    v_mean = torch.Tensor([0.485, 0.456, 0.406])[None, ..., None, None]
    v_std = torch.Tensor([0.229, 0.224, 0.225])[None, ..., None, None]

    for name in tqdm(names, desc='testing'):
        img = Image.open(os.path.join(img_root, name+'.jpg')).convert('RGB')
        ww, hh = img.size
        img = np.array(img, dtype=np.uint8)
        img = img.transpose((2, 0, 1))
        img = (torch.from_numpy(img[None, ...]).float() - v_mean)/v_std

        net.feats = []
        x = net(img.cuda())
        net.feats += [x]
        feat=torch.cat((F.upsample(net.feats[1], size=(hh, ww)), F.upsample(net.feats[2], size=(hh, ww))), 1).detach().cpu().numpy()
        feat = feat.transpose((0, 2, 3, 1))
        proc_feat(name, feat, img_root, prob_root, output_root)


    print('done')
    fm, mae, _, _ = fm_and_mae(output_root, '/home/zeng/data/datasets/saliency_Dataset/ECSSD/masks')
    print(fm)
    print(mae)

