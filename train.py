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

from datasets import ImageNetDetCls, Folder
from models import FCN
from evaluate import fm_and_mae

from tqdm import tqdm
import random

random.seed(1996)


home = os.path.expanduser("~")

parser = argparse.ArgumentParser()
parser.add_argument('--train_dir', default='%s/data/datasets/ILSVRC2014_devkit' % home)  # training dataset
parser.add_argument('--val_dir', default='%s/data/datasets/saliency_Dataset/ECSSD' % home)  # training dataset
parser.add_argument('--base', default='densenet169')  # batch size
parser.add_argument('--b', type=int, default=36)  # batch size
parser.add_argument('--e', type=int, default=100)  # epoches
opt = parser.parse_args()
print(opt)

name = 'ClsMax_%s'%opt.base
check_dir = '../ROTSfiles/' + name

if not os.path.exists(check_dir):
    os.mkdir(check_dir)

img_size = 256

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

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


def validate(loader, net, output_dir, gt_dir):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    net.eval()
    loader = tqdm(loader, desc='validating')
    for ib, (data, lbl, img_name, w, h) in enumerate(loader):
        with torch.no_grad():
            outputs, _, _ = net(Variable(data.cuda(), volatile=True))
        outputs = F.sigmoid(outputs).data.squeeze(1).cpu().numpy()
        outputs *= 255
        for ii, msk in enumerate(outputs):
            msk = Image.fromarray(msk.astype(np.uint8))
            msk = msk.resize((w[ii], h[ii]))
            msk.save('{}/{}.png'.format(output_dir, img_name[ii]), 'PNG')
    fm, mae, _, _ = fm_and_mae(output_dir, gt_dir)
    net.train()
    return fm, mae


def main():


    # data
    train_loader = torch.utils.data.DataLoader(
        ImageNetDetCls(opt.train_dir,
                    transforms.Compose([transforms.Resize((img_size, img_size)),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=mean, std=std)])),
        batch_size=opt.b, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        Folder(opt.val_dir,
               source_transform=transforms.Compose([transforms.Resize((img_size, img_size))]),
               target_transform=transforms.Compose([transforms.Resize((img_size, img_size))]),
               mean=mean, std=std),
        batch_size=opt.b, shuffle=False, num_workers=4, pin_memory=True)
    # models
    net = FCN(base=opt.base)
    net = net.cuda()
    net.train()
    # net = nn.DataParallel(net).cuda()
    optimizer = torch.optim.Adam([
        {'params': net.parameters(), 'lr': 1e-4},
    ])
    # if not 'resnet' in opt.base:
    #     optimizer = torch.optim.Adam([
    #         {'params': net.parameters(), 'lr': 1e-4},
    #     ])
    # else:
    #     base_lr = 5e-3
    #     optimizer = torch.optim.SGD([
    #         {'params': [param for name, param in net.named_parameters() if name[-4:] == 'bias'],
    #          'lr': 2 * base_lr},
    #         {'params': [param for name, param in net.named_parameters() if name[-4:] != 'bias'],
    #          'lr': base_lr, 'weight_decay': 1e-4},
    #     ], momentum=0.9, nesterov=True)
    logs = {'best_it': 0, 'best': 0}
    train_iter = iter(train_loader)
    i_data = 0
    for it in tqdm(range(10000)):
        if i_data >= len(train_loader):
            train_iter = iter(train_loader)
            i_data = 0
        data, lbl = train_iter.next()
        i_data += 1
        big_msk, msk, cls = net(Variable(data.cuda()))
        loss = F.binary_cross_entropy_with_logits(cls, Variable(lbl.cuda())) + \
               5e-4*F.binary_cross_entropy(msk, torch.zeros(msk.shape).cuda())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # if 'resnet' in opt.base:
        #     lr = base_lr * ((1 - float(it) / 10000) ** 0.9)
        #     optimizer = torch.optim.SGD([
        #         {'params': [param for name, param in net.named_parameters() if name[-4:] == 'bias'],
        #          'lr': 2 * lr},
        #         {'params': [param for name, param in net.named_parameters() if name[-4:] != 'bias'],
        #          'lr': lr, 'weight_decay': 1e-4},
        #     ], momentum=0.9, nesterov=True)
        # visualize
        if it % 10 == 0:
            big_msk = F.sigmoid(big_msk).expand(-1, 3, -1, -1)
            writer.add_scalar('C_global', loss.data[0], it)
            writer.add_image('msk', torchvision.utils.make_grid(big_msk.data[:6]), it)
            image = make_image_grid(data[:6], mean, std)
            writer.add_image('Image', torchvision.utils.make_grid(image), it)
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
    main()