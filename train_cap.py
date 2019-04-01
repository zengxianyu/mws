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
from datasets import CocoCaption, Folder
from models import EncDec
from evaluate import fm_and_mae

from tqdm import tqdm
import random

random.seed(1996)


home = os.path.expanduser("~")

parser = argparse.ArgumentParser()
parser.add_argument('--train_dir', default='%s/data/datasets/coco' % home)  # training dataset
parser.add_argument('--val_dir', default='%s/data/datasets/saliency_Dataset/ECSSD' % home)  # training dataset
parser.add_argument('--base', default='densenet169')  # batch size
parser.add_argument('--b', type=int, default=24)  # batch size
parser.add_argument('--e', type=int, default=100)  # epoches
opt = parser.parse_args()
print(opt)

name = 'Cap_%s'%opt.base

img_size = 256

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

# tensorboard writer
os.system('rm -rf ./runs_%s/*'%name)
if not os.path.exists('./runs_%s'%name):
    os.mkdir('./runs_%s'%name)
writer = SummaryWriter('./runs_%s/'%name + datetime.now().strftime('%B%d  %H:%M:%S'))
print('sb')


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
            outputs = net(Variable(data.cuda(), volatile=True))
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

    check_dir = '../ROTS2files/' + name

    if not os.path.exists(check_dir):
        os.mkdir(check_dir)

    # data
    # Load vocabulary wrapper
    with open('vocab.pkl', 'rb') as f:
        vocab = pickle.load(f)
    train_loader = torch.utils.data.DataLoader(
        CocoCaption(os.path.join(opt.train_dir, 'images/train2014'),
                    os.path.join(opt.train_dir, 'annotations/captions_train2014.json'),
                    vocab,
                    transforms.Compose([transforms.Resize((img_size, img_size)),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=mean, std=std)])),
        batch_size=opt.b, shuffle=True, num_workers=4, pin_memory=True, collate_fn=caption_collate_fn)
    val_loader = torch.utils.data.DataLoader(
        Folder(opt.val_dir,
               source_transform=transforms.Compose([transforms.Resize((img_size, img_size))]),
               target_transform=transforms.Compose([transforms.Resize((img_size, img_size))]),
               mean=mean, std=std),
        batch_size=opt.b, shuffle=False, num_workers=4, pin_memory=True)
    # models
    net = EncDec(len(vocab))
    net = net.cuda()
    # net.load_state_dict(torch.load('../ROTS2files/Cap_densenet169/net-iter400.pth'))
    net.train()
    optimizer = torch.optim.Adam([
        {'params': net.parameters(), 'lr': 1e-4},
    ])
    logs = {'best_it': 0, 'best': 0}
    it = 0
    for epoch in tqdm(range(opt.e)):
        # if epoch >= 10:
        #     optimizer.param_groups[0]['lr'] *= 0.1
        print('--------------------epoch %d-----------------------'%epoch)
        net.train()
        for i, (images, captions, lengths) in enumerate(train_loader):
            # if num_e+i > 1500 and (num_e+i)%100 == 0:
            #     optimizer.param_groups[0]['lr'] *= 0.1
            images = images.cuda()
            captions = captions.cuda()
            targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]

            # Forward, backward and optimize
            big_msk, msk, outputs = net(images, captions, lengths)
            loss = F.cross_entropy(outputs, targets) + \
                   5e-4*F.binary_cross_entropy(msk, torch.zeros(msk.shape).cuda())
                    # 5e-4*F.l1_loss(msk, torch.zeros(msk.shape).cuda())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            it += 1
            # visualize
            if it % 10 == 0:
                big_msk = F.sigmoid(big_msk).expand(-1, 3, -1, -1)
                writer.add_scalar('C_global', loss.item(), it)
                writer.add_image('msk', torchvision.utils.make_grid(big_msk.data[:6]), it)
                image = make_image_grid(images[:6], mean, std)
                writer.add_image('Image', torchvision.utils.make_grid(image), it)
            if it != 0 and it % 200 == 0:
                fm, mae = validate(val_loader, net, os.path.join(check_dir, 'results'),
                                os.path.join(opt.val_dir, 'masks'))
                print(u'损失: %.4f'%(loss.item()))
                print(u'最大FM: iteration %d的%.4f, 这次FM: %.4f'%(logs['best_it'], logs['best'], fm))
                logs[epoch] = {'FM': fm}
                if fm > logs['best']:
                    logs['best'] = fm
                    logs['best_it'] = it
                    torch.save(net.state_dict(), '%s/net-iter%d.pth' % (check_dir, it))
                with open(os.path.join(check_dir, 'logs.json'), 'w') as outfile:
                    json.dump(logs, outfile)


if __name__ == "__main__":
    main()