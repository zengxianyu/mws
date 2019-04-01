import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
import torch.nn.functional as F
import numpy as np
import pdb
from .densenet import *
from .resnet import *
from .vgg import *
import sys
thismodule = sys.modules[__name__]


dim_dict = {
    'resnet101': [512, 1024, 2048],
    'resnet152': [512, 1024, 2048],
    'resnet50': [512, 1024, 2048],
    'resnet34': [128, 256, 512],
    'resnet18': [128, 256, 512],
    'densenet121': [256, 512, 1024],
    'densenet161': [384, 1056, 2208],
    'densenet169': [64, 128, 256, 640, 1664],
    # 'densenet169': [256, 640, 1664],
    'densenet201': [256, 896, 1920],
    'vgg': [256, 512, 512]
}


def get_upsampling_weight(in_channels, out_channels, kernel_size):
    """Make a 2D bilinear kernel suitable for upsampling"""
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size),
                      dtype=np.float64)
    weight[range(in_channels), range(out_channels), :, :] = filt
    return torch.from_numpy(weight).float()


class ParamPool(nn.Module):
    def __init__(self, input_c):
        super(ParamPool, self).__init__()
        self.conv = nn.Conv2d(input_c, 1, kernel_size=1, bias=False)

    def forward(self, x):
        bsize, c, ssize, _ = x.shape
        w = self.conv(x)
        w = F.softmax(w.view(bsize, 1, -1), 2)
        w = w.view(bsize, 1, ssize, ssize)
        x = (x*w).sum(3).sum(2)
        return x


def proc_densenet(model):
    def hook(module, input, output):
        model.feats[output.device.index] += [output]
    model.features.transition3[-2].register_forward_hook(hook)
    model.features.transition2[-2].register_forward_hook(hook)
    # dilation
    def remove_sequential(all_layers, network):
        for layer in network.children():
            if isinstance(layer, nn.Sequential):  # if sequential layer, apply recursively to layers in sequential layer
                remove_sequential(all_layers, layer)
            if list(layer.children()) == []:  # if leaf node, add it to list
                all_layers.append(layer)
    model.features.transition3[-1].kernel_size = 1
    model.features.transition3[-1].stride = 1
    all_layers = []
    remove_sequential(all_layers, model.features.denseblock4)
    for m in all_layers:
        if isinstance(m, nn.Conv2d) and m.kernel_size==(3, 3):
            m.dilation = (2, 2)
            m.padding = (2, 2)
    return model


procs = {
    'densenet169': proc_densenet,
    'densenet201': proc_densenet,
         }


class EncoderCNN(nn.Module):
    def __init__(self, embed_size, patt_size=512, base='densenet169'):
        """Load the pretrained ResNet-152 and replace top fc layer."""
        super(EncoderCNN, self).__init__()
        if 'vgg' in base:
            dims = dim_dict['vgg'][::-1]
        else:
            dims = dim_dict[base][::-1]
        self.preds = nn.ModuleList([nn.Conv2d(d, 1, kernel_size=1) for d in dims])
        self.upscales = nn.ModuleList([
            nn.ConvTranspose2d(1, 1, 4, 2, 1),
            nn.ConvTranspose2d(1, 1, 4, 2, 1),
            nn.ConvTranspose2d(1, 1, 16, 8, 4),
        ])
        self.linear = nn.Linear(patt_size, embed_size)
        self.reduce = nn.Conv2d(dims[0], patt_size, kernel_size=3, padding=1)
        self.param_pool = ParamPool(patt_size)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 0.02)
                if m.bias is not None:
                    m.bias.data.fill_(0)
            if isinstance(m, nn.ConvTranspose2d):
                assert m.kernel_size[0] == m.kernel_size[1]
                initial_weight = get_upsampling_weight(
                    m.in_channels, m.out_channels, m.kernel_size[0])
                m.weight.data.copy_(initial_weight)

        self.feature = getattr(thismodule, base)(pretrained=True)
        self.feature.feats = {}
        self.feature = procs[base](self.feature)
        for m in self.feature.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.requires_grad=False

    def forward(self, x):
        """Extract feature vectors from input images."""
        # pdb.set_trace()
        # with torch.no_grad():
        #     features = self.resnet(images)
        # features = features.reshape(features.size(0), -1)
        # features = self.bn(self.linear(features))
        # return features
        self.feature.feats[x.device.index] = []
        x = self.feature(x)
        feats = self.feature.feats[x.device.index]
        feats += [x]
        feats = feats[::-1]
        msk = self.preds[0](feats[0])
        big_msk = msk
        # big_msk = self.upscales[0](msk)
        # big_msk = F.upsample(msk, scale_factor=16)
        msk = F.sigmoid(msk)
        msk_feat = self.reduce(feats[0])*msk
        # msk_feat = F.avg_pool2d(msk_feat, 16).squeeze(3).squeeze(2)
        msk_feat = self.param_pool(msk_feat)
        msk_feat = self.linear(msk_feat)
        big_msk = F.upsample_bilinear(big_msk, scale_factor=16)
        if self.training:
            return big_msk, msk, msk_feat
        else:
            return big_msk


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
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True) 
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


class EncDec(nn.Module):
    def __init__(self, vocab_size, embed_size=256, hidden_size=512, num_layers=1, max_seq_length=20):
        super(EncDec, self).__init__()
        self.encoder = EncoderCNN(embed_size)
        self.decoder = DecoderRNN(embed_size, hidden_size, vocab_size, num_layers, max_seq_length)

    def forward(self, images, captions=None, lengths=None):
        if self.training:
            if captions is not None:
                big_msk, msk, msk_feat = self.encoder(images)
                outputs = self.decoder(msk_feat, captions, lengths)
                return big_msk, msk, outputs
            else:
                big_msk, msk, msk_feat = self.encoder(images)
                return big_msk, msk
        else:
            return self.encoder(images)

