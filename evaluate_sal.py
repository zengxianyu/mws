from __future__ import print_function
import numpy as np
import os
import PIL.Image as Image
import pdb
from multiprocessing import Pool
from functools import partial
import matplotlib.pyplot as plt
eps = np.finfo(float).eps


def print_table():
    base_dir = '/home/zeng/data/datasets/saliency_Dataset'
    algs = ['Ours-Seg', 'Ours-Seg-woSeg', 'Ours-N1-Seg', 'Ours-N1-Seg-woSeg']
    datasets = ['ECSSD']
    # datasets = ['ECSSD', 'PASCALS', 'HKU-IS', 'DUT-test']
    # algs = ['Ours-Seg-crf']
    for alg in algs:
        print(alg+'& ', end='')
        for i, dset in enumerate(datasets):
            input_dir = '{}/results/{}-Sal/{}'.format(base_dir, dset, alg)
            gt_dir = '{}/{}/masks'.format(base_dir, dset)
            output_dir = '{}/results/{}-npy'.format(base_dir, dset)
            if os.path.exists(os.path.join(output_dir, alg+'.npz')):
                sb = np.load(os.path.join(output_dir, alg+'.npz'))
                maxfm, mae = sb['maxfm'], sb['mea']
            else:
                maxfm, mae, _, _ = fm_and_mae(input_dir, gt_dir, output_dir, alg)
            if i != len(datasets)-1:
                print('%.3f&%.3f& '%(round(maxfm, 3), round(mae, 3)), end='')
            else:
                print('%.3f&%.3f\\\\'%(round(maxfm, 3), round(mae, 3)), end='\n')
                print('\hline', end='\n')


def draw_curves():
    base_dir = '/home/zeng/data/datasets/saliency_Dataset'
    algs = ['Ours-Seg', 'Ours-Seg-woSeg', 'Ours-N1-Seg', 'Ours-N1-Seg-woSeg', 'DSS']
    datasets = ['ECSSD']
    # color = iter(plt.cm.rainbow(np.linspace(0, 1, len(algs))))
    for dset in datasets:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for i, alg in enumerate(algs):
            sb = np.load('{}/results/{}-npy/{}.npz'.format(base_dir, dset, alg))
            ax.plot(sb['recs'], sb['pres'], linewidth=2, label=alg)
        ax.grid(True)
        ax.set_xlabel('Recall', fontsize=14)
        ax.set_ylabel('Precision', fontsize=14)
        handles, labels = ax.get_legend_handles_labels()
        lgd = ax.legend(handles, labels, loc='center left', bbox_to_anchor=(0.5, -0.5), ncol=8, fontsize=14)
        fig.savefig('%s.pdf'%dset, bbox_extra_artists=(lgd,), bbox_inches='tight')


def eva_one(param):
    input_name, gt_name = param
    mask = Image.open(input_name)
    gt = Image.open(gt_name)
    mask = mask.resize(gt.size)
    mask = np.array(mask, dtype=np.float)
    if len(mask.shape) != 2:
        mask = mask[:, :, 0]
    mask = (mask - mask.min()) / (mask.max()-mask.min()+eps)
    gt = np.array(gt, dtype=np.uint8)
    if len(gt.shape)>2:
        gt = gt[:, :, 0]
    gt[gt != 0] = 1
    pres = []
    recs = []
    mea = np.abs(gt-mask).mean()
    # threshold fm
    binary = np.zeros(mask.shape)
    th = 2*mask.mean()
    if th > 1:
        th = 1
    binary[mask >= th] = 1
    sb = (binary * gt).sum()
    pre = sb / (binary.sum()+eps)
    rec = sb / (gt.sum()+eps)
    thfm = 1.3 * pre * rec / (0.3 * pre + rec + eps)
    for th in np.linspace(0, 1, 21):
        binary = np.zeros(mask.shape)
        binary[ mask >= th] = 1
        pre = (binary * gt).sum() / (binary.sum()+eps)
        rec = (binary * gt).sum() / (gt.sum()+ eps)
        pres.append(pre)
        recs.append(rec)
    pres = np.array(pres)
    recs = np.array(recs)
    return thfm, mea, recs, pres


def fm_and_mae(input_dir, gt_dir, output_dir=None, name=None):
    if output_dir is not None and not os.path.exists(output_dir):
        os.mkdir(output_dir)

    filelist_gt = os.listdir(gt_dir)
    gt_format = filelist_gt[0].split('.')[-1]
    filelist_gt = ['.'.join(f.split('.')[:-1]) for f in filelist_gt]

    filelist_pred = os.listdir(input_dir)
    pred_format = filelist_pred[0].split('.')[-1]
    filelist_pred = ['.'.join(f.split('.')[:-1]) for f in filelist_pred]

    filelist = list(set(filelist_gt)&set(filelist_pred))

    inputlist = [os.path.join(input_dir, '.'.join([_name, pred_format])) for _name in filelist]
    gtlist = [os.path.join(gt_dir, '.'.join([_name, gt_format])) for _name in filelist]

    pool = Pool(4)
    results = pool.map(eva_one, zip(inputlist, gtlist))
    thfm, m_mea, m_recs, m_pres = list(map(list, zip(*results)))
    m_mea = np.array(m_mea).mean()
    m_pres = np.array(m_pres).mean(0)
    m_recs = np.array(m_recs).mean(0)
    thfm = np.array(thfm).mean()
    fms = 1.3 * m_pres * m_recs / (0.3*m_pres + m_recs + eps)
    maxfm = fms.max()
    if not (output_dir is None or name is None):
        np.savez('%s/%s.npz'%(output_dir, name), mea=m_mea, thfm=thfm, maxfm = maxfm, recs=m_recs, pres=m_pres, fms=fms)
    return maxfm, m_mea, m_recs, m_pres


if __name__ == '__main__':
    # fm, mae, _, _ = fm_and_mae('/home/crow/WSLfiles/WTCW_woSeg_densenet169/results',
    #                      '/home/crow/data/datasets/saliency_Dataset/ECSSD/masks')
    # print(fm)
    # print(mae)
    print_table()
    # draw_curves()



