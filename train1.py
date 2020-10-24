# coding=utf-8
import pdb
from PIL import Image
import numpy as np
import time
import torch
import torch.nn as nn
import torchvision
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F
from logger import Logger
from tqdm import tqdm
import sys
from networks.salcap import SalCap
from networks.salcls import SalCls
from datasets import CocoCaption, caption_collate_fn, Folder
from datasets import ImageNetDetCls, Folder
from datasets import ImageList
from evaluate_sal import fm_and_mae
from datasets.build_vocab import Vocabulary
import pickle
import json
import os
import random
import argparse
from train_capsal import visualize as visualize_cap
from train_clssal import visualize as visualize_cls
import nltk


class TransLoss(nn.Module):
    def forward(self, x, target):
        target = target.clone()
        _,_,h,w = target.size()
        th = target.view(-1, 1, h*w).mean(2, keepdim=True)
        th = th[..., None]*2
        #th = 0.5
        target = (target>th).float()
        loss = F.binary_cross_entropy(x, target)
        return loss, target



if __name__ == "__main__":
    nltk.download('punkt')

    parser = argparse.ArgumentParser(description='train cap sal')
    parser.add_argument('--pathsave', default='./output/debug_1', type=str)
    parser.add_argument('--pathimg_train_cap', default='./data/train2014', type=str)
    parser.add_argument('--pathann_train_cap', default='./data/captions_train2014.json', type=str)
    parser.add_argument('--pathimg_train_cls', default='./data/ILSVRC/Data/DET/train', type=str)
    parser.add_argument('--pathann_train_cls', default='./data/ILSVRC2014_devkit', type=str)
    parser.add_argument('--pathimg_val_sal', default='./data/ECSSD/images', type=str)
    parser.add_argument('--pathann_val_sal', default='./data/ECSSD/masks', type=str)
    parser.add_argument('--pathimg', default='./data/images', type=str)
    parser.add_argument('--pathimglst', default='./data/list_images.txt', type=str)
    parser.add_argument('--batchsize', default=8, type=int)
    parser.add_argument('--numstart', default=-1, type=int)
    parser.add_argument('--numprint', default=200, type=int)
    parser.add_argument('--numval', default=2000, type=int)
    parser.add_argument('--numtrain', default=400000, type=int)
    parser.add_argument('--numworkers', default=8, type=int)
    parser.add_argument('--wr', default=5e-3, type=float)
    parser.add_argument('--wt', default=1e-2, type=float)
    args = parser.parse_args()


    mean = torch.Tensor([0.485, 0.456, 0.406]).cuda()[None, ..., None, None]
    std = torch.Tensor([0.229, 0.224, 0.225]).cuda()[None, ..., None, None]

    n_vis = 2

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

    with open("{}_capsal.txt".format(args.pathsave), "w") as f:
        for k,v in vars(args).items():
            line = "{}: {}\n".format(k,v)
            print(line)
            f.write(line)

    with open('vocab_trainval.pkl', 'rb') as f:
        vocab = pickle.load(f)

    sal_val_loader = torch.utils.data.DataLoader(
        Folder(args.pathimg_val_sal, args.pathann_val_sal,
               crop=None, flip=False, rotate=None, size=256,
               mean=None, std=None, training=False),
        batch_size=args.batchsize, shuffle=False, num_workers=args.numworkers, pin_memory=True)
    cap_train_loader = torch.utils.data.DataLoader(
        CocoCaption(args.pathimg_train_cap, args.pathann_train_cap, vocab,
                    crop=None, flip=True, rotate=None, size=256,
                    mean=None, std=None, training=True),
        batch_size=args.batchsize, shuffle=True, num_workers=args.numworkers, pin_memory=True, collate_fn=caption_collate_fn)
    id2w = cap_train_loader.dataset.vocab.idx2word
    id2w = np.array(id2w)
    visualize_cap.id2w = id2w
    cls_train_loader = torch.utils.data.DataLoader(
        ImageNetDetCls(args.pathimg_train_cls, args.pathann_train_cls,
                    crop=None, flip=True, rotate=None, size=256,
                    mean=None, std=None, training=True),
        batch_size=args.batchsize, shuffle=True, num_workers=args.numworkers, pin_memory=True)
    img_train_loader = torch.utils.data.DataLoader(
        ImageList(args.pathimg, args.pathimglst,
                    crop=None, flip=True, rotate=None, size=256,
                    mean=None, std=None, training=True),
        batch_size=args.batchsize, shuffle=True, num_workers=args.numworkers, pin_memory=True)
    #cap_val_loader = torch.utils.data.DataLoader(
    #    CocoCaption(args.pathimg_val, args.pathann_val, vocab,
    #                crop=None, flip=True, rotate=None, size=256,
    #                mean=None, std=None, training=True),
    #    batch_size=args.batchsize, shuffle=True, num_workers=args.numworkers, pin_memory=True, collate_fn=caption_collate_fn)
    dict_train_loader = {'cls': cls_train_loader, 'cap': cap_train_loader, 
            'img': img_train_loader}

    netc = SalCls(n_class=200)
    netc.cuda()
    netp = SalCap(vocab_size=len(cap_train_loader.dataset.vocab))
    netp.cuda()
    netp.load_state_dict(torch.load("./5e-3_19999_ca.pth"))
    netc.load_state_dict(torch.load("./5e-4_49999_cl.pth"))

    if args.numstart > 0:
        state = torch.load(f"{pathchk}/{args.numstart}.pth.tar")
        netc.load_state_dict(state['cls'])
        netp.load_state_dict(state['cap'])

    param = list(filter(lambda p: p.requires_grad, netc.parameters())) +\
            list(filter(lambda p: p.requires_grad, netp.parameters()))
    optimizer = torch.optim.Adam(param, lr=1e-4)

    def val_sal(num):
        netc.eval()
        netc.rcap = False
        netp.eval()
        netp.rcap = False
        for i, (img, names, WWs, HHs) in tqdm(enumerate(sal_val_loader), desc="val"):
            with torch.no_grad():
                img = (img.cuda()-mean)/std
                msk_big1 = netc(img)
                msk_big1 = F.sigmoid(msk_big1)
                msk_big2 = netp(img)
                msk_big2 = F.sigmoid(msk_big2)
                msk_big = (msk_big1+msk_big2)/2
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
        netp.train()
        netc.train()
        return maxfm, mae

    def validate(num):
        log = {}
        fm, mae = val_sal(num)
        writer.add_scalar("valfm", fm, num)
        writer.add_scalar("valmae", mae, num)
        log['fm'] = fm
        log['mae'] = mae
        #val_loss = val_cap(num)
        #log['valloss'] = val_loss
        #writer.add_scalar("valloss", val_loss, num)
        return log

    logs = {}
    netc.train()
    netp.train()
    dict_train_iter = {}
    dict_it = {}
    for k,v in dict_train_loader.items(): dict_train_iter[k] = iter(v)
    for k,v in dict_train_loader.items(): dict_it[k] = 0
    ttloss = TransLoss()

    def get_batchd(key):
        if dict_it[key] >= len(dict_train_loader[key]):
            dict_train_iter[key] = iter(dict_train_loader[key])
            dict_it[key] = 0
        data = dict_train_iter[key].next()
        dict_it[key] += 1
        return data

    if args.numstart > 0:
        log = validate(args.numstart)
        writer.write_html()
        logs[args.numstart] = log
        with open(args.pathsave+".json", "w") as f:
            json.dump(logs, f)

    for num in range(args.numstart+1, args.numtrain):
        optimizer.zero_grad()
        ### cap
        img, captions, lengths = get_batchd('cap')
        captions = captions.cuda()
        img = (img.cuda()-mean)/std
        targets = pack_padded_sequence(captions, lengths, batch_first=True)
        netp.rcap=True
        netc.rcap=True
        big_msk, msk, outputs = netp(img, captions, lengths)
        big_msk = F.sigmoid(big_msk)
        loss = F.cross_entropy(outputs, targets.data) + \
                args.wr*F.binary_cross_entropy(F.sigmoid(msk), 
                        torch.zeros_like(msk, device=msk.device))

        pgt = big_msk.detach()
        pmsk, _, _ = netc(img)
        pmsk = F.sigmoid(pmsk)
        if num > -1:#200:
            _l, _g = ttloss(pmsk, pgt)
            loss += (_l*args.wt)
            #loss += ttloss(pmsk, pgt)*args.wt
        loss.backward()
        if num % args.numprint == 0:
            # visualize
            _, cappred = outputs.max(1)
            targets.data.data = cappred.detach().data
            cappred, _ = pad_packed_sequence(targets, batch_first=True)
            visualize_cap(img[:n_vis]*std+mean, cappred[:n_vis],
                    captions[:n_vis], big_msk[:n_vis], loss.item(), num, writer)
            print(f"train iteration {num} | caption loss {loss.item()}")

        ### cls
        img, onehot = get_batchd('cls')
        onehot = onehot.cuda()
        img = (img.cuda()-mean)/std
        big_msk, msk, outputs = netc(img)
        big_msk = F.sigmoid(big_msk)
        loss = F.binary_cross_entropy_with_logits(outputs, onehot) + \
                args.wr*F.binary_cross_entropy(F.sigmoid(msk), 
                        torch.zeros_like(msk, device=msk.device))

        pgt = big_msk.detach()
        netp.rcap = False
        pmsk = netp(img)
        pmsk = F.sigmoid(pmsk)
        if num > -1:#200:
            #loss += ttloss(pmsk, pgt)*args.wt
            _l, _g = ttloss(pmsk, pgt)
            loss += (_l*args.wt)
        loss.backward()
        if num % args.numprint == 0:
            # visualize
            onehot_pred = (F.sigmoid(outputs).detach()>0.5)
            visualize_cls(img[:n_vis]*std+mean, onehot_pred[:n_vis],
                    onehot[:n_vis], big_msk[:n_vis], loss.item(), num, writer)
            _g = _g[:n_vis]
            _g = torch.cat((_g, _g, _g), 1)
            writer.add_single_image("dbg", torchvision.utils.make_grid(_g), num)
            writer.write_html()
            print(f"train iteration {num} | class loss {loss.item()}")

        ### img
        #img = get_batchd('img')

        optimizer.step()

        if (num+1) % args.numval == 0:
            state = {'cls':netc.state_dict(), 'cap': netp.state_dict()}
            torch.save(state, pathchk+f"/{num}.pth.tar")
            log = validate(num)
            writer.write_html()
            logs[num] = log
            with open(args.pathsave+".json", "w") as f:
                json.dump(logs, f)
