# coding=utf-8
import pdb
from PIL import Image
import numpy as np
import time
import torch
import torchvision
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F
from logger import Logger
from tqdm import tqdm
import sys
from networks.salcap import SalCap
from datasets import CocoCaption, caption_collate_fn, Folder
from evaluate_sal import fm_and_mae
from datasets.build_vocab import Vocabulary
import pickle
import json
import os
import random
import argparse
import nltk
nltk.download('punkt')



def visualize(img, cappred, captions, big_msk, loss, i, writer):
    writer.add_single_image("image_cap", torchvision.utils.make_grid(img), i)
    big_msk = torch.cat([big_msk, big_msk, big_msk], 1)
    writer.add_single_image("mask_cap", torchvision.utils.make_grid(big_msk), i)
    list_t = tokentensor2text(captions)
    writer.add_text("caption", list_t, i)
    list_t = tokentensor2text(cappred)
    writer.add_text("cappred", list_t, i)
    writer.add_scalar("loss_cap", loss, i)

def tokentensor2text(tokens):
    text = visualize.id2w[tokens.cpu().numpy()]
    list_t = [' '.join(t) for t in text]
    return list_t

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='train cap sal')
    parser.add_argument('--pathsave', default='./output/debug_capsal', type=str)
    parser.add_argument('--pathimg_train', default='./data/train2014', type=str)
    parser.add_argument('--pathann_train', default='./data/captions_train2014.json', type=str)
    #parser.add_argument('--pathimg_val', default='./data/train2014', type=str)
    #parser.add_argument('--pathann_val', default='./data/captions_train2014.json', type=str)
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
        CocoCaption(args.pathimg_train, args.pathann_train, vocab,
                    crop=None, flip=True, rotate=None, size=256,
                    mean=None, std=None, training=True),
        batch_size=args.batchsize, shuffle=True, num_workers=args.numworkers, pin_memory=True, collate_fn=caption_collate_fn)
    id2w = cap_train_loader.dataset.vocab.idx2word
    id2w = np.array(id2w)
    visualize.id2w = id2w


    #cap_val_loader = torch.utils.data.DataLoader(
    #    CocoCaption(args.pathimg_val, args.pathann_val, vocab,
    #                crop=None, flip=True, rotate=None, size=256,
    #                mean=None, std=None, training=True),
    #    batch_size=args.batchsize, shuffle=True, num_workers=args.numworkers, pin_memory=True, collate_fn=caption_collate_fn)

    net = SalCap(vocab_size=len(cap_train_loader.dataset.vocab))
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

    def val_cap(num):
        net.eval()
        net.rcap=True
        total_loss = 0
        n_sample = 0
        N_val = 5000//args.batchsize
        val_iter = iter(cap_val_loader)
        N_val = min(N_val, len(val_iter))
        for i in tqdm(range(N_val), desc="val"):
            img, captions, lengths = val_iter.next()
            captions = captions.cuda()
            with torch.no_grad():
                img = (img.cuda()-mean)/std
                targets = pack_padded_sequence(captions, lengths, batch_first=True)
                _, _, outputs = net(img, captions, lengths)
                loss = F.cross_entropy(outputs, targets.data)
                total_loss += loss.item()
        total_loss /= N_val
        print(f"val iteration {num} | caption loss {total_loss}")
        net.train()
        return total_loss

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
    net.train()
    cap_train_iter = iter(cap_train_loader)
    it = 0
    if args.numstart > 0:
        log = validate(args.numstart)
        writer.write_html()
        logs[args.numstart] = log
        with open(args.pathsave+".json", "w") as f:
            json.dump(logs, f)

    for num in range(args.numstart+1, args.numtrain):
        if it >= len(cap_train_loader):
            cap_train_iter = iter(cap_train_loader)
            it = 0
        img, captions, lengths = cap_train_iter.next()
        it += 1
        captions = captions.cuda()
        img = (img.cuda()-mean)/std
        targets = pack_padded_sequence(captions, lengths, batch_first=True)
        net.rcap=True
        big_msk, msk, outputs = net(img, captions, lengths)
        big_msk = F.sigmoid(big_msk)
        loss = F.cross_entropy(outputs, targets.data) + \
                args.wr*F.binary_cross_entropy(F.sigmoid(msk), 
                        torch.zeros_like(msk, device=msk.device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if num % args.numprint == 0:
            # visualize
            _, cappred = outputs.max(1)
            targets.data.data = cappred.detach().data
            cappred, _ = pad_packed_sequence(targets, batch_first=True)
            visualize(img[:4]*std+mean, cappred[:4],
                    captions[:4], big_msk[:4], loss.item(), num, writer)
            writer.write_html()
            print(f"train iteration {num} | caption loss {loss.item()}")
        if (num+1) % args.numval == 0:
            torch.save(net.state_dict(), pathchk+f"/{num}.pth")
            log = validate(num)
            writer.write_html()
            logs[num] = log
            with open(args.pathsave+".json", "w") as f:
                json.dump(logs, f)
