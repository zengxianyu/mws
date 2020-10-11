import torch.nn as nn
import torch
import torch.nn.functional as F
from .tools import mr, deep_mr
import pdb


class SelfTrainLoss(nn.Module):

    def __init__(self, vmean, vstd):
        super(SelfTrainLoss, self).__init__()
        self.vstd = vstd.cuda()
        self.vmean = vmean.cuda()

    def forward(self, images, masks):
        assert len(masks.shape)==4
        arr_imgs = (images*self.vstd+self.vmean).cpu().numpy().transpose((0, 2, 3, 1))
        arr_masks = masks.squeeze(1).detach().cpu().numpy()
        gt_mr = mr(arr_imgs, [arr_masks])
        gt_mr = torch.Tensor(gt_mr).unsqueeze(1).float()
        loss = F.binary_cross_entropy(masks, gt_mr.cuda())
        return loss


class CoTrainLoss(nn.Module):

    def __init__(self, vmean, vstd):
        super(CoTrainLoss, self).__init__()
        self.vstd = vstd
        self.vmean = vmean

    def forward(self, images, list_masks):
        arr_imgs = (images*self.vstd+self.vmean).cpu().numpy().transpose((0, 2, 3, 1))
        arr_gts = [masks.squeeze(1).detach().cpu().numpy() for masks in list_masks]
        gt_mr = mr(arr_imgs, arr_gts)
        gt_mr = torch.Tensor(gt_mr).unsqueeze(1).float().cuda()
        self.gt_mr = gt_mr
        loss1 = F.binary_cross_entropy(list_masks[0], gt_mr)
        loss2 = F.binary_cross_entropy(list_masks[1], gt_mr)
        return loss1+loss2


class DeepCoTrainLoss(nn.Module):

    def __init__(self, vmean, vstd, net_cls, net_cap):
        super(DeepCoTrainLoss, self).__init__()
        self.vstd = vstd
        self.vmean = vmean
        self.net_cls = net_cls
        self.net_cap = net_cap

    def forward(self, images, list_masks):
        _, _, h, w = images.shape
        feats_cls = self.net_cls.module.feature.feats
        feats_cls = torch.cat([torch.cat([feats_cls[i][j].detach().cpu() for i in range(torch.cuda.device_count())], 0)
                               for j in range(len(feats_cls[0]))], 1)

        feats_cap = self.net_cap.module.encoder.feature.feats
        feats_cap = torch.cat([torch.cat([feats_cap[i][j].detach().cpu() for i in range(torch.cuda.device_count())], 0)
                               for j in range(len(feats_cap[0]))], 1)
        feats = torch.cat((feats_cls, feats_cap), 1)
        arr_feats = feats.numpy().transpose((0, 2, 3, 1))
        arr_imgs = (images*self.vstd+self.vmean).cpu().numpy().transpose((0, 2, 3, 1))
        arr_gts = [masks.squeeze(1).detach().cpu().numpy() for masks in list_masks]
        gt_mr = deep_mr(arr_imgs, arr_feats, arr_gts)
        gt_mr = torch.Tensor(gt_mr).unsqueeze(1).float().cuda()
        self.gt_mr = gt_mr
        loss1 = F.binary_cross_entropy(list_masks[0], gt_mr)
        loss2 = F.binary_cross_entropy(list_masks[1], gt_mr)
        return loss1+loss2


class TransTrainLoss(nn.Module):

    def __init__(self, vmean, vstd):
        super(TransTrainLoss, self).__init__()
        self.vstd = vstd.cuda()
        self.vmean = vmean.cuda()

    def forward(self, images, masks, gt):
        assert len(masks.shape)==4
        arr_imgs = (images*self.vstd+self.vmean).cpu().numpy().transpose((0, 2, 3, 1))
        arr_gt = gt.squeeze(1).detach().cpu().numpy()
        gt_mr = mr(arr_imgs, [arr_gt])
        gt_mr = torch.Tensor(gt_mr).unsqueeze(1).float()
        loss = F.binary_cross_entropy(masks, gt_mr.cuda())
        return loss
