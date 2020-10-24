# coding=utf-8
import pdb
from PIL import Image
import numpy as np
import time
import torch
import torchvision
import torch.nn.functional as F
from logger import Logger
from tqdm import tqdm
import sys
from networks.salcls import SalCls
from datasets import ImageNetDetCls, Folder
from evaluate_sal import fm_and_mae
import json
import scipy.io as sio
import os
import random
import argparse

path_synsets = "./data/ILSVRC2014_devkit/data/meta_det.mat"
synsets = sio.loadmat(path_synsets)['synsets'][0]
id2w = []
for item in synsets:
    id2w.append(item[2][0]+',')
pad = np.array(['']*len(id2w))
id2w = np.array(id2w)
id2w = np.stack((pad, id2w), 1)
index1 = np.arange(200)[None, ...]


def onehottensor2text(onehot):
    text = id2w[index1, onehot.cpu().int().numpy()]
    list_t = [''.join(t) for t in text]
    return list_t

def visualize(img, onehot_pred, onehot, big_msk, loss, i, writer):
    writer.add_single_image("image_cls", torchvision.utils.make_grid(img), i)
    big_msk = torch.cat([big_msk, big_msk, big_msk], 1)
    writer.add_single_image("mask_cls", torchvision.utils.make_grid(big_msk), i)
    list_t = onehottensor2text(onehot)
    writer.add_text("gt_cls", list_t, i, n_word=20)
    list_t = onehottensor2text(onehot_pred)
    writer.add_text("pred_cls", list_t, i, n_word=20)
    writer.add_scalar("loss_cls", loss, i)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='train cls sal')
    parser.add_argument('--pathsave', default='./output/debug_clssal', type=str)
    parser.add_argument('--pathimg_train', default='./data/ILSVRC/Data/DET/train', type=str)
    parser.add_argument('--pathann_train', default='./data/ILSVRC2014_devkit', type=str)
    #parser.add_argument('--pathimg_val', default='./data/ILSVRC/Data/DET/val', type=str)
    #parser.add_argument('--pathann_val', default='./data/ILSVRC2014_devkit', type=str)
    parser.add_argument('--pathimg_val_sal', default='./data/ECSSD/images', type=str)
    parser.add_argument('--pathann_val_sal', default='./data/ECSSD/masks', type=str)
    parser.add_argument('--batchsize', default=8, type=int)
    parser.add_argument('--numstart', default=-1, type=int)
    parser.add_argument('--numprint', default=200, type=int)
    parser.add_argument('--numval', default=2000, type=int)
    parser.add_argument('--numtrain', default=400000, type=int)
    parser.add_argument('--numworkers', default=4, type=int)
    parser.add_argument('--wr', default=5e-4, type=float)
    args = parser.parse_args()


    mean = torch.Tensor([0.485, 0.456, 0.406]).cuda()[None, ..., None, None]
    std = torch.Tensor([0.229, 0.224, 0.225]).cuda()[None, ..., None, None]

    if not os.path.exists(args.pathsave):
        os.mkdir(args.pathsave)

    def append_dir(name):
        pathappend = args.pathsave + "/" + name
        if not os.path.exists(pathappend):
            os.mkdir(pathappend)
        return pathappend

    pathvrst = append_dir("val_output")
    pathchk = append_dir("checkpoints")
    pathlog = append_dir("pages")
    writer = Logger(pathlog)

    with open("{}_clssal.txt".format(args.pathsave), "w") as f:
        for k,v in vars(args).items():
            line = "{}: {}\n".format(k,v)
            print(line)
            f.write(line)

    sal_val_loader = torch.utils.data.DataLoader(
        Folder(args.pathimg_val_sal, args.pathann_val_sal,
               crop=None, flip=False, rotate=None, size=256,
               mean=None, std=None, training=False),
        batch_size=args.batchsize, shuffle=False, num_workers=args.numworkers, pin_memory=True)
    cls_train_loader = torch.utils.data.DataLoader(
        ImageNetDetCls(args.pathimg_train, args.pathann_train,
                    crop=None, flip=True, rotate=None, size=256,
                    mean=None, std=None, training=True),
        batch_size=args.batchsize, shuffle=True, num_workers=args.numworkers, pin_memory=True)

    net = SalCls(n_class=200)
    net.cuda()

    if args.numstart > 0:
        net.load_state_dict(torch.load(f"{pathchk}/{args.numstart}.pth"))
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=1e-4)

    def val_sal(num):
        net.eval()
        net.rcap = False
        for i, (img, names, WWs, HHs) in tqdm(enumerate(sal_val_loader), desc="val"):
            with torch.no_grad():
                img = (img.cuda()-mean)/std
                msk_big = net(img)
                msk_big = F.sigmoid(msk_big)
            msk_big = msk_big.squeeze(1)
            msk_big = msk_big.cpu().numpy()*255
            for b, _msk in enumerate(msk_big):
                name = names[b]
                WW = WWs[b]
                HH = HHs[b]
                _msk = Image.fromarray(_msk.astype(np.uint8))
                _msk = _msk.resize((WW, HH))
                _msk.save(f"{pathvrst}/{name}.png")
        maxfm, mae, _, _ = fm_and_mae(pathvrst, args.pathann_val_sal)
        print(f"val iteration {num} | FM {maxfm} | MAE {mae}")
        net.train()
        return maxfm, mae


    def validate(num):
        log = {}
        fm, mae = val_sal(num)
        writer.add_scalar("valfm", fm, num)
        writer.add_scalar("valmae", mae, num)
        log['fm'] = fm
        log['mae'] = mae
        return log

    logs = {}
    net.train()
    cls_train_iter = iter(cls_train_loader)
    it = 0
    if args.numstart > 0:
        log = validate(args.numstart)
        writer.write_html()
        logs[args.numstart] = log
        with open(args.pathsave+".json", "w") as f:
            json.dump(logs, f)

    for num in range(args.numstart+1, args.numtrain):
        if it >= len(cls_train_loader):
            cls_train_iter = iter(cls_train_loader)
            it = 0
        img, onehot  = cls_train_iter.next()
        it += 1
        onehot = onehot.cuda()
        img = (img.cuda()-mean)/std
        net.rcap=True
        big_msk, msk, outputs = net(img)
        big_msk = F.sigmoid(big_msk)
        loss = F.binary_cross_entropy_with_logits(outputs, onehot) + \
                args.wr*F.binary_cross_entropy(F.sigmoid(msk), 
                        torch.zeros_like(msk, device=msk.device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if num % args.numprint == 0:
            # visualize
            onehot_pred = (F.sigmoid(outputs).detach()>0.5)
            visualize(img[:4]*std+mean, onehot_pred[:4],
                    onehot[:4], big_msk[:4], loss.item(), num, writer)
            writer.write_html()
            print(f"train iteration {num} | caption loss {loss.item()}")
        if (num+1) % args.numval == 0:
            torch.save(net.state_dict(), pathchk+f"/{num}.pth")
            log = validate(num)
            writer.write_html()
            logs[num] = log
            with open(args.pathsave+".json", "w") as f:
                json.dump(logs, f)

