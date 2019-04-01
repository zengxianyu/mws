# coding=utf-8
import torch
import torchvision
from torchvision import transforms
import torch.nn.functional as F
import torch.nn as nn
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

from datasets import WSFolder, Folder
from models import FCN0, FCN, EncDec, DeepLab
from evaluate import fm_and_mae
from datasets.build_vocab import Vocabulary
import pickle

from tqdm import tqdm
import random

random.seed(1996)


home = os.path.expanduser("~")

parser = argparse.ArgumentParser()
parser.add_argument('--train_dir', default='%s/data/datasets/saliency_Dataset/DUT-train' % home)  # training dataset
# parser.add_argument('--sal_dir', default='%s/data/datasets/dut_syn' % home)  # training dataset
parser.add_argument('--self_dir', default='%s/ROTS2files/DUT-train_two_mr2_crf_bin' % home)  # training dataset
parser.add_argument('--val_dir', default='%s/data/datasets/saliency_Dataset/ECSSD' % home)  # training dataset
parser.add_argument('--base', default='densenet169')  # training dataset
parser.add_argument('--img_size', type=int, default=256)  # batch size
parser.add_argument('--b', type=int, default=26)  # batch size
parser.add_argument('--max', type=int, default=3000)  # epoches
opt = parser.parse_args()
print(opt)


def make_image_grid(img, mean, std):
    img = make_grid(img)
    for i in range(3):
        img[i] *= std[i]
        img[i] += mean[i]
    return img


def validate(loader, net, output_dir, gt_dir):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    net.eval()
    loader = tqdm(loader, desc='validating')
    for ib, (data, lbl, img_name, w, h) in enumerate(loader):
        with torch.no_grad():
            outputs, _, _ = net(data.cuda())
            outputs = F.sigmoid(outputs)
        outputs = outputs.squeeze(1).cpu().numpy()
        outputs *= 255
        for ii, msk in enumerate(outputs):
            msk = Image.fromarray(msk.astype(np.uint8))
            msk = msk.resize((w[ii], h[ii]))
            msk.save('{}/{}.png'.format(output_dir, img_name[ii]), 'PNG')
    fm, mae, _, _= fm_and_mae(output_dir, gt_dir)
    net.train()
    return fm, mae


def main():

    check_dir = '../ROTS2files/' + name

    if not os.path.exists(check_dir):
        os.mkdir(check_dir)

    # data
    train_loader = torch.utils.data.DataLoader(
        WSFolder(opt.train_dir, opt.self_dir,
               crop=0.9, flip=True,
               source_transform=transforms.Compose([transforms.Resize((opt.img_size, opt.img_size))]),
               target_transform=transforms.Compose([transforms.Resize((opt.img_size, opt.img_size))]),
               mean=mean, std=std, training=True),
        batch_size=opt.b, shuffle=True, num_workers=4, pin_memory=True)
    # train_loader = torch.utils.data.DataLoader(
    #     Folder(opt.train_dir,
    #              crop=0.9, flip=True,
    #              source_transform=transforms.Compose([transforms.Resize((opt.img_size, opt.img_size))]),
    #              target_transform=transforms.Compose([transforms.Resize((opt.img_size, opt.img_size))]),
    #              mean=mean, std=std, training=True),
    #     batch_size=opt.b, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        Folder(opt.val_dir,
               source_transform=transforms.Compose([transforms.Resize((opt.img_size, opt.img_size))]),
               target_transform=transforms.Compose([transforms.Resize((opt.img_size, opt.img_size))]),
               mean=mean, std=std),
        batch_size=opt.b, shuffle=False, num_workers=4, pin_memory=True)
    # models
    # Load vocabulary wrapper
    net = DeepLab(base=opt.base, c_output=1)
    net = nn.DataParallel(net).cuda()
    net.load_state_dict(torch.load('../ROTS2files/Deeplab_dutmr4_densenet169/net-iter400.pth'))
    net.train()
    optimizer = torch.optim.Adam([
        {'params': net.parameters(), 'lr': 1e-5},
    ])
    # optimizer = torch.optim.SGD([
    #     {'params': net.parameters(), 'lr': 2.5e-4, 'momentum': 0.9,'weight_decay': 0.0005},
    # ])
    logs = {'best_it':0, 'best': 0}
    sal_data_iter = iter(train_loader)
    i_sal_data = 0
    for it in tqdm(range(opt.max)):
        if it > 1000 and it % 100 == 0:
            # deeplab
            optimizer.param_groups[0]['lr'] *= 0.5
        # ================train on pixel-level label==================
        if i_sal_data >= len(train_loader):
            sal_data_iter = iter(train_loader)
            i_sal_data = 0
        data, lbl, _ = sal_data_iter.next()
        i_sal_data += 1
        data = data.cuda()
        lbl = lbl.unsqueeze(1).cuda()
        msk, _, _ = net(data)
        # loss = F.binary_cross_entropy_with_logits(msk, lbl)
        self_gt = F.sigmoid(msk)
        self_gt[self_gt>0.5] = 1
        self_gt[self_gt<=0.5] = 0

        loss = 0.95*F.binary_cross_entropy_with_logits(msk, lbl) +\
            0.05 * F.binary_cross_entropy_with_logits(msk, self_gt.detach())
        # loss = F.binary_cross_entropy_with_logits(msk, lbl)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if it % 10 == 0:
            writer.add_scalar('loss', loss.item(), it)
            image = make_image_grid(data[:6], mean, std)
            writer.add_image('Image', torchvision.utils.make_grid(image), it)
            big_msk = F.sigmoid(msk).expand(-1, 3, -1, -1)
            writer.add_image('msk', torchvision.utils.make_grid(big_msk.data[:6]), it)
            big_msk = lbl.expand(-1, 3, -1, -1)
            writer.add_image('gt', torchvision.utils.make_grid(big_msk.data[:6]), it)
        # if it % 200 == 0:
        if it != 0 and it % 200 == 0:
            fm, mae = validate(val_loader, net, os.path.join(check_dir, 'results'),
                                              os.path.join(opt.val_dir, 'masks'))
            print(u'损失: %.4f'%(loss.item()))
            print(u'最大FM: iteration %d的%.4f, 这次FM: %.4f'%(logs['best_it'], logs['best'], fm))
            logs[it] = {'FM': fm}
            if fm > logs['best']:
                logs['best'] = fm
                logs['best_it'] = it
                torch.save(net.state_dict(), '%s/net-best.pth' % (check_dir))
            with open(os.path.join(check_dir, 'logs.json'), 'w') as outfile:
                json.dump(logs, outfile)
            torch.save(net.state_dict(), '%s/net-iter%d.pth' % (check_dir, it))


if __name__ == "__main__":
    for ttag in range(3, 10):

        name = 'Deeplab_dutmr4_{}_{}'.format(ttag, opt.base)

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        # mean = [103.939, 116.779, 123.68]

        # tensorboard writer
        os.system('rm -rf ./runs_%s/*'%name)
        writer = SummaryWriter('./runs_%s/'%name + datetime.now().strftime('%B%d  %H:%M:%S'))
        if not os.path.exists('./runs_%s'%name):
            os.mkdir('./runs_%s'%name)
        main()