# coding=utf-8
import torch
import torchvision
from torchvision import transforms
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision.utils import make_grid
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from datetime import datetime
import os
import pdb
import numpy as np
from PIL import Image
import argparse
import json
import pickle
from datasets.build_vocab import Vocabulary
from datasets.coco import caption_collate_fn
from datasets import CocoCaption, Folder, ImageNetDetCls, ImageFiles
from models import EncDec, FCN
from evaluate import fm_and_mae

from tqdm import tqdm
import random
from test_sp_avg_mr import mr_func

random.seed(1996)


home = os.path.expanduser("~")

parser = argparse.ArgumentParser()
parser.add_argument('--cap_train_dir', default='%s/data/datasets/coco' % home)  # training dataset
parser.add_argument('--semi_train_dir', default='%s/data/datasets/ILSVRC12VOC/images' % home)  # training dataset
parser.add_argument('--cls_train_dir', default='%s/data/datasets/ILSVRC2014_devkit' % home)  # training dataset
parser.add_argument('--val_dir', default='%s/data/datasets/saliency_Dataset/ECSSD' % home)  # training dataset
parser.add_argument('--base', default='densenet169')  # batch size
parser.add_argument('--b', type=int, default=32)  # batch size
parser.add_argument('--e', type=int, default=100)  # epoches
opt = parser.parse_args()
print(opt)

name = 'Two2_mr_%s'%opt.base

img_size = 256
msk_size = 16
xx, yy = np.meshgrid(np.arange(msk_size), np.arange(msk_size))
val = np.sqrt((yy.astype(np.float)-msk_size/2)**2 + (xx.astype(np.float)-msk_size/2)**2)
val = np.exp(-0.001*val)
val = torch.Tensor(val).float()
val0 = val.max()-val
val1 = val - val.min()

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

vmean = torch.Tensor(mean)[None, ..., None, None]
vstd = torch.Tensor(std)[None, ..., None, None]

# tensorboard writer
os.system('rm -rf ./runs_%s/*'%name)
writer = SummaryWriter('./runs_%s/'%name + datetime.now().strftime('%B%d  %H:%M:%S'))
if not os.path.exists('./runs_%s'%name):
    os.mkdir('./runs_%s'%name)


def make_image_grid(img, mean, std):
    img = make_grid(img)
    for i in range(3):
        img[i] *= std[i]
        img[i] += mean[i]
    return img


# def validate(loader, net_cls, net_cap, output_dir, gt_dir):
#     if not os.path.exists(output_dir):
#         os.mkdir(output_dir)
#     net_cls.eval()
#     net_cap.eval()
#     loader = tqdm(loader, desc='validating')
#     for ib, (data, lbl, img_name, w, h) in enumerate(loader):
#         with torch.no_grad():
#             outputs_cls, _, _ = net_cls(data.cuda(0))
#             outputs_cap = net_cap(data.cuda(1))
#         outputs = (F.sigmoid(outputs_cls.cpu()+2) + F.sigmoid(outputs_cap.cpu()+2))/2
#         outputs = outputs.squeeze(1).cpu().numpy()
#         outputs *= 255
#         for ii, msk in enumerate(outputs):
#             msk = Image.fromarray(msk.astype(np.uint8))
#             msk = msk.resize((w[ii], h[ii]))
#             msk.save('{}/{}.png'.format(output_dir, img_name[ii]), 'PNG')
#     fm, mae = fm_and_mae(output_dir, gt_dir)
#     net_cls.train()
#     net_cap.train()
#     return fm, mae


def validate(loader, net_cls, net_cap, output_dir, gt_dir):
    if not os.path.exists(output_dir+'_cls'):
        os.mkdir(output_dir+'_cls')
    if not os.path.exists(output_dir+'_cap'):
        os.mkdir(output_dir+'_cap')
    net_cls.eval()
    net_cap.eval()
    loader = tqdm(loader, desc='validating')
    for ib, (data, lbl, img_name, w, h) in enumerate(loader):
        with torch.no_grad():
            outputs, _, _ = net_cls(data.cuda(0))
        outputs = F.sigmoid(outputs).data.squeeze(1).cpu().numpy()
        outputs *= 255
        for ii, msk in enumerate(outputs):
            msk = Image.fromarray(msk.astype(np.uint8))
            msk = msk.resize((w[ii], h[ii]))
            msk.save('{}/{}.png'.format(output_dir+'_cls', img_name[ii]), 'PNG')
        with torch.no_grad():
            outputs = net_cap(data.cuda(1))
        outputs = F.sigmoid(outputs).data.squeeze(1).cpu().numpy()
        outputs *= 255
        for ii, msk in enumerate(outputs):
            msk = Image.fromarray(msk.astype(np.uint8))
            msk = msk.resize((w[ii], h[ii]))
            msk.save('{}/{}.png'.format(output_dir+'_cap', img_name[ii]), 'PNG')
    fm_cls, mae_cls, _, _ = fm_and_mae(output_dir+'_cls', gt_dir)
    fm_cap, mae_cap, _, _ = fm_and_mae(output_dir+'_cap', gt_dir)
    net_cls.train()
    net_cap.train()
    return fm_cls, mae_cls, fm_cap, mae_cap


def main():

    check_dir = '../ROTS2files/' + name

    if not os.path.exists(check_dir):
        os.mkdir(check_dir)

    # data
    # Load vocabulary wrapper
    with open('vocab.pkl', 'rb') as f:
        vocab = pickle.load(f)
    cls_train_loader = torch.utils.data.DataLoader(
        ImageNetDetCls(opt.cls_train_dir,
                       transforms.Compose([transforms.Resize((img_size, img_size)),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize(mean=mean, std=std)])),
        batch_size=opt.b, shuffle=True, num_workers=4, pin_memory=True)
    cap_train_loader = torch.utils.data.DataLoader(
        CocoCaption(os.path.join(opt.cap_train_dir, 'images/train2014'),
                    os.path.join(opt.cap_train_dir, 'annotations/captions_train2014.json'),
                    vocab,
                    transforms.Compose([transforms.Resize((img_size, img_size)),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=mean, std=std)])),
        batch_size=opt.b, shuffle=True, num_workers=4, pin_memory=True, collate_fn=caption_collate_fn)
    semi_train_loader = torch.utils.data.DataLoader(
        ImageFiles(root=opt.semi_train_dir, crop=0.9, flip=True,
                   source_transform=transforms.Compose([transforms.Resize((img_size, img_size))]),
                   mean=mean, std=std, training=True),
        batch_size=opt.b, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        Folder(opt.val_dir,
               source_transform=transforms.Compose([transforms.Resize((img_size, img_size))]),
               target_transform=transforms.Compose([transforms.Resize((img_size, img_size))]),
               mean=mean, std=std),
        batch_size=opt.b, shuffle=False, num_workers=4, pin_memory=True)
    # models
    cls_net = FCN(base=opt.base)
    cls_net = cls_net.cuda(0)
    # cls_net.load_state_dict(torch.load('./cls-two.pth'))
    # cls_net.load_state_dict(torch.load('../ROTS2files/Two_wo_Undensenet169/cls-iter1600.pth'))
    cls_net.train()
    cap_net = EncDec(len(vocab))
    cap_net = cap_net.cuda(1)
    # cap_net.load_state_dict(torch.load('./cap-two.pth'))
    # cap_net.load_state_dict(torch.load('../ROTS2files/Two_wo_Undensenet169/cap-iter1600.pth'))
    cap_net.train()
    optimizer = torch.optim.Adam([
        {'params': cls_net.parameters(), 'lr': 1e-4},
        {'params': cap_net.parameters(), 'lr': 1e-4},
    ])
    logs = {'cls':{'best_it': 0, 'best': 0}, 'cap':{'best_it':0, 'best': 0}}
    cls_iter = iter(cls_train_loader)
    cap_iter = iter(cap_train_loader)
    semi_iter = iter(semi_train_loader)
    i_cls = 0
    i_cap = 0
    i_semi = 0
    reg_val = 5e-4
    for it in tqdm(range(10000)):
        if it >= 4000 and it % 100 == 0:
            reg_val *= 0.5
        """classification"""
        if i_cls >= len(cls_train_loader):
            cls_iter = iter(cls_train_loader)
            i_cls = 0
        images_cls, lbl = cls_iter.next()
        i_cls+= 1
        big_msk_cls, msk, cls = cls_net(images_cls.cuda(0))
        cls_loss = F.binary_cross_entropy_with_logits(cls, lbl.cuda(0)) + \
            reg_val*F.binary_cross_entropy(msk, torch.zeros(msk.shape).cuda(0))
        optimizer.zero_grad()
        cls_loss.backward()

        temp_gt = msk.detach().cuda(1)
        temp_gt[temp_gt>0.5] = 1
        temp_gt[temp_gt<=0.5] = 0
        big_msk_temp, msk_temp = cap_net(images_cls.cuda(1))
        aux_loss = 0.01*F.binary_cross_entropy(msk_temp, temp_gt)
        aux_loss.backward()
        """caption"""
        if i_cap >= len(cap_train_loader):
            cap_iter = iter(cap_train_loader)
            i_cap = 0
        images_cap, captions, lengths = cap_iter.next()
        i_cap += 1
        targets = pack_padded_sequence(captions.cuda(1), lengths, batch_first=True)[0]
        big_msk_cap, msk, outputs = cap_net(images_cap.cuda(1), captions.cuda(1), lengths)
        cap_loss = F.cross_entropy(outputs, targets) + \
                   reg_val*F.binary_cross_entropy(msk, torch.zeros(msk.shape).cuda(1))
        cap_loss.backward()

        big_msk_temp, msk_temp, _ = cls_net(images_cap.cuda(0))
        temp_gt = msk.detach().cuda(0)
        temp_gt[temp_gt>0.5] = 1
        temp_gt[temp_gt<=0.5] = 0
        aux_loss = 0.01*F.binary_cross_entropy(msk_temp, temp_gt)
        aux_loss.backward()
        # visualize
        if it % 10 == 0:
            writer.add_scalar('CLS', cls_loss.item(), it)
            image = make_image_grid(images_cls[:6], mean, std)
            writer.add_image('ImageCLS', torchvision.utils.make_grid(image), it)
            big_msk = F.sigmoid(big_msk_cls).expand(-1, 3, -1, -1)
            writer.add_image('mskCLS', torchvision.utils.make_grid(big_msk.data[:6]), it)

            writer.add_scalar('CAP', cap_loss.item(), it)
            image = make_image_grid(images_cap[:6], mean, std)
            writer.add_image('ImageCAP', torchvision.utils.make_grid(image), it)
            big_msk = F.sigmoid(big_msk_cap).expand(-1, 3, -1, -1)
            writer.add_image('mskCAP', torchvision.utils.make_grid(big_msk.data[:6]), it)
        """unlabeled"""
        if i_semi >= len(semi_train_loader):
            semi_iter = iter(semi_train_loader)
            i_semi = 0
        images = semi_iter.next()
        i_semi += 1
        bsize = images.size(0)
        big_msk_cls, msk_cls, _ = cls_net(images.cuda(0))
        big_msk_cap, msk_cap = cap_net(images.cuda(1))

        # temp_gt = torch.zeros(bsize, 1, 16, 16)
        # th = msk_cap.cpu().mean()
        # temp_gt[msk_cap.cpu()>th] = 1
        # aux_loss = 0.01*F.binary_cross_entropy(msk_cls, temp_gt.cuda(0))
        # aux_loss.backward()
        # temp_gt = torch.zeros(bsize, 1, 16, 16)
        # th = msk_cls.cpu().mean()
        # temp_gt[msk_cls.cpu()>th] = 1
        # aux_loss = 0.01*F.binary_cross_entropy(msk_cap, temp_gt.cuda(1))
        # aux_loss.backward()

        big_msk_cls = F.sigmoid(big_msk_cls)
        big_msk_cap = F.sigmoid(big_msk_cap)
        arr_imgs = (images*vstd+vmean).numpy().transpose((0, 2, 3, 1))
        arr_cls = big_msk_cls.squeeze(1).detach().cpu().numpy()
        arr_cap = big_msk_cap.squeeze(1).detach().cpu().numpy()
        gt_mr = mr_func(arr_imgs, arr_cls, arr_cap)
        gt_mr = torch.Tensor(gt_mr).unsqueeze(1).float()
        aux_loss = 0.01*F.binary_cross_entropy(big_msk_cls, gt_mr.cuda(0))
        aux_loss.backward()
        aux_loss = 0.01*F.binary_cross_entropy(big_msk_cap, gt_mr.cuda(1))
        aux_loss.backward()

        optimizer.step()
        if it != 0 and it % 100 == 0:
            # _, mae_cls, _, mae_cap = validate(val_loader, cls_net, cap_net, os.path.join(check_dir, 'results'),
            #                 os.path.join(opt.val_dir, 'masks'))
            fm_cls, _, fm_cap, _ = validate(val_loader, cls_net, cap_net, os.path.join(check_dir, 'results'),
                            os.path.join(opt.val_dir, 'masks'))
            print(u'cls损失: %.4f'%(cls_loss.item()))
            print(u'cap损失: %.4f'%(cap_loss.item()))
            print(u'分类最大FM: iteration %d的%.4f, 这次FM: %.4f'%(logs['cls']['best_it'], logs['cls']['best'], fm_cls))
            logs['cls'][it] = {'fm': fm_cls}
            print(u'说明最大FM: iteration %d的%.4f, 这次FM: %.4f'%(logs['cap']['best_it'], logs['cap']['best'], fm_cap))
            logs['cap'][it] = {'fm': fm_cap}
            if fm_cls > logs['cls']['best']:
                logs['cls']['best'] = fm_cls
                logs['cls']['best_it'] = it
            if fm_cap > logs['cap']['best']:
                logs['cap']['best'] = fm_cap
                logs['cap']['best_it'] = it
            with open(os.path.join(check_dir, 'logs.json'), 'w') as outfile:
                json.dump(logs, outfile)
            torch.save(cls_net.state_dict(), '%s/cls-iter%d.pth' % (check_dir, it))
            torch.save(cap_net.state_dict(), '%s/cap-iter%d.pth' % (check_dir, it))


if __name__ == "__main__":
    main()