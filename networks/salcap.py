import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
import torch.nn.functional as F
import pdb
from .densenet import *
from .tools import fraze_bn, weight_init, dim_dict
from .parampool import ParamPool
import numpy as np
import sys


def proc_densenet(model):
    def hook(module, input, output):
        model.feats[output.device.index] += [output]
    model.features.transition2[-2].register_forward_hook(hook)
    model.features.transition1[-2].register_forward_hook(hook)
    # dilation
    # def remove_sequential(all_layers, network):
    #     for layer in network.children():
    #         if isinstance(layer, nn.Sequential):  # if sequential layer, apply recursively to layers in sequential layer
    #             remove_sequential(all_layers, layer)
    #         if list(layer.children()) == []:  # if leaf node, add it to list
    #             all_layers.append(layer)
    model.features.transition3[-1].kernel_size = 1
    model.features.transition3[-1].stride = 1
    # all_layers = []
    # remove_sequential(all_layers, model.features.denseblock4)
    # for m in all_layers:
    #     if isinstance(m, nn.Conv2d) and m.kernel_size==(3, 3):
    #         m.dilation = (2, 2)
    #         m.padding = (2, 2)
    model.classifier = None
    return model


procs = {
    'densenet169': proc_densenet,
         }

class EncoderCNN(nn.Module):
    def __init__(self, patt_size=512, base='densenet169', usem=False):
        super(EncoderCNN, self).__init__()
        dims = dim_dict[base][::-1]
        self.pred = nn.Conv2d(dims[0], 1, kernel_size=1)
        self.reduce = nn.Conv2d(dims[0], patt_size, kernel_size=3, padding=1)
        self.msk_size = 16
        self.param_pool = ParamPool(patt_size)
        self.apply(weight_init)

        #self.feature = getattr(thismodule, base)(pretrained=True)
        self.feature = densenet169(pretrained=True)
        self.feature.feats = {}
        self.feature = procs[base](self.feature)
        self.usem = usem

        self.apply(fraze_bn)

    def forward(self, x):
        """Extract feature vectors from input images."""
        self.feature.feats[x.device.index] = []
        feats0 = self.feature(x)
        msk = self.pred(feats0)
        big_msk = F.upsample(msk, scale_factor=16, mode='bilinear')
        feat = self.reduce(feats0)
        if self.usem:
            msk_feat = self.param_pool(feat*F.sigmoid(msk))
        else:
            msk_feat = self.param_pool(feat)
        return big_msk, msk, msk_feat


class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers, max_seq_length=20):
        """Set the hyper-parameters and build the layers."""
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.max_seg_length = max_seq_length

    def forward(self, features, captions, lengths):
        """Decode image feature vectors and generates captions."""
        embeddings = self.embed(captions)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        l = len(lengths)//torch.cuda.device_count()
        s = l*torch.cuda.current_device()
        packed = pack_padded_sequence(embeddings, lengths[s:s+l], batch_first=True)
        hiddens, _ = self.lstm(packed)
        outputs = self.linear(hiddens[0])
        return outputs

    def sample(self, features, states=None):
        """Generate captions for given image features using greedy search."""
        sampled_ids = []
        inputs = features.unsqueeze(1)
        for i in range(self.max_seg_length):
            hiddens, states = self.lstm(inputs, states)          # hiddens: (batch_size, 1, hidden_size)
            outputs = self.linear(hiddens.squeeze(1))            # outputs:  (batch_size, vocab_size)
            _, predicted = outputs.max(1)                        # predicted: (batch_size)
            sampled_ids.append(predicted)
            inputs = self.embed(predicted)                       # inputs: (batch_size, embed_size)
            inputs = inputs.unsqueeze(1)                         # inputs: (batch_size, 1, embed_size)
        sampled_ids = torch.stack(sampled_ids, 1)                # sampled_ids: (batch_size, max_seq_length)
        return sampled_ids


class SalCap(nn.Module):
    def __init__(self, vocab_size, base='densenet169',
                 embed_size=512, hidden_size=512, num_layers=1, max_seq_length=20, usem=False, rcap=True):
        super(SalCap, self).__init__()
        self.encoder = EncoderCNN(embed_size, base=base, usem=usem)
        self.decoder = DecoderRNN(embed_size, hidden_size, vocab_size, num_layers, max_seq_length)
        self.rcap= rcap

    def forward(self, images, captions=None, lengths=None):
        big_msk, msk, msk_feat = self.encoder(images)
        if self.rcap:
            outputs = self.decoder(msk_feat, captions, lengths)
            return big_msk, msk, outputs
        else:
            return big_msk
