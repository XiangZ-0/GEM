import math
import torch
import numpy as np
from torch import nn
from torch.nn import init
import torch.nn.functional as F

def init_weights(net, init_type='normal', init_gain=0.02):
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)
    net.apply(init_func)  # apply the initialization function <init_func>

class up(nn.Module):
    def __init__(self, inChannels, outChannels):
        super(up, self).__init__()
        self.conv1 = nn.Conv2d(inChannels, outChannels, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(outChannels, outChannels, 3, stride=1, padding=1)

    def forward(self, x, skpCn):
        x = F.interpolate(x, scale_factor=2, mode="bicubic")
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x + skpCn))
        return x 

class down(nn.Module):
    def __init__(self, inChannels, outChannels, filterSize):
        super(down, self).__init__()
        self.conv1 = nn.Conv2d(inChannels,outChannels,filterSize,stride=1,padding=int((filterSize - 1) / 2))
        self.conv2 = nn.Conv2d(outChannels,outChannels,filterSize,stride=1,padding=int((filterSize - 1) / 2))

    def forward(self, x):
        x = F.avg_pool2d(x, 2)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return x 

class Encoder(nn.Module):
    def __init__(self, inChannels):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(inChannels, 16, 7, stride=1, padding=3)
        self.conv2 = nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.down1 = down(16, 32, 5)
        self.down2 = down(32, 64, 3)
        self.down3 = down(64, 128, 3)
        self.down4 = down(128, 128, 3)
    
    def forward(self, x):
        b,c,h,w = x.shape
        
        x = F.relu(self.conv1(x))
        s1 = F.relu(self.conv2(x))
        s2 = self.down1(s1)
        s3 = self.down2(s2)
        s4 = self.down3(s3)
        s5 = self.down4(s4)

        return s1, s2, s3, s4, s5

class Decoder(nn.Module):
    def __init__(self, outChannels, ends_with_relu=False):
        super(Decoder, self).__init__()
        self._ends_with_relu = ends_with_relu
        self.up1 = up(128, 128) 
        self.up2 = up(128, 64) 
        self.up3 = up(64, 32) 
        self.up4 = up(32, 16)
        self.conv = nn.Conv2d(16, outChannels, 3, stride=1, padding=1)
        
    def forward(self, s1, s2, s3, s4, s5):
        x = self.up1(s5, s4)
        x = self.up2(x, s3)
        x = self.up3(x, s2)
        x = self.up4(x, s1)

        if self._ends_with_relu == True:
            x = F.relu(self.conv(x))
        else:
            x = self.conv(x)

        return x

class Basic_MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_list): 
        super().__init__()
        layers = []
        lastv = in_dim
        for hidden in hidden_list:
            layers.append(nn.Linear(lastv, hidden))
            layers.append(nn.ReLU())
            lastv = hidden
        layers.append(nn.Linear(lastv, out_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        shape = x.shape[:-1]
        x = self.layers(x.view(-1, x.shape[-1]))
        return x.view(*shape, -1)

## the following codes are modified based on LIIF project (https://github.com/yinboc/liif)
class MLP(nn.Module):
    def __init__(self, inChannel, outChannel, hidden_list=None,
                 local_ensemble=True, feat_unfold=False, cell_decode=True, double_inChannels=True):
        super().__init__()
        self.local_ensemble = local_ensemble
        self.feat_unfold = feat_unfold
        self.cell_decode = cell_decode
        
        if double_inChannels:
            imnet_in_dim = inChannel * 2
        else:
            imnet_in_dim = inChannel
        if self.feat_unfold:
            imnet_in_dim *= 9
        imnet_in_dim += 2 # attach coord
        if self.cell_decode:
            imnet_in_dim += 2
        
        self.imnet = Basic_MLP(imnet_in_dim, outChannel, hidden_list)
        
    def query_rgb(self, inp1, coord, cell1, inp2):
        feat1 = inp1
        feat2 = inp2

        if self.feat_unfold:
            feat1 = F.unfold(feat1, 3, padding=1).view(
                feat1.shape[0], feat1.shape[1] * 9, feat1.shape[2], feat1.shape[3])
            feat2 = F.unfold(feat2, 3, padding=1).view(
                feat2.shape[0], feat2.shape[1] * 9, feat2.shape[2], feat2.shape[3])

        if self.local_ensemble:
            vx_lst = [-1, 1]
            vy_lst = [-1, 1]
            eps_shift = 1e-6
        else:
            vx_lst, vy_lst, eps_shift = [0], [0], 0

        # field radius (global: [-1, 1])
        rx = 2 / feat1.shape[-2] / 2
        ry = 2 / feat1.shape[-1] / 2

        feat1_coord = make_coord(feat1.shape[-2:], flatten=False) \
            .permute(2, 0, 1) \
            .unsqueeze(0).expand(feat1.shape[0], 2, *feat1.shape[-2:]).to(feat1.device)

        preds = []
        areas = []
        for vx in vx_lst:
            for vy in vy_lst:
                coord_ = coord.clone()
                coord_[:, :, 0] += vx * rx + eps_shift
                coord_[:, :, 1] += vy * ry + eps_shift
                coord_.clamp_(-1 + 1e-6, 1 - 1e-6)
                
                q_feat1 = F.grid_sample(
                    feat1, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)
                q_coord1 = F.grid_sample(
                    feat1_coord, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)
                rel_coord1 = coord - q_coord1
                rel_coord1[:, :, 0] *= feat1.shape[-2]
                rel_coord1[:, :, 1] *= feat1.shape[-1]
                inp1 = torch.cat([q_feat1, rel_coord1], dim=-1)

                if self.cell_decode:
                    rel_cell1 = cell1.clone()
                    rel_cell1[:, :, 0] *= feat1.shape[-2]
                    rel_cell1[:, :, 1] *= feat1.shape[-1]
                    inp1 = torch.cat([inp1, rel_cell1], dim=-1)

                q_feat2 = F.grid_sample(
                    feat2, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)
                inp = torch.cat([inp1, q_feat2], dim=-1)
                
                bs, q = coord.shape[:2]
                pred = self.imnet(inp.view(bs * q, -1)).view(bs, q, -1)
                preds.append(pred)

                area = torch.abs(rel_coord1[:, :, 0] * rel_coord1[:, :, 1])
                areas.append(area + 1e-9)

        tot_area = torch.stack(areas).sum(dim=0)
        if self.local_ensemble:
            t = areas[0]; areas[0] = areas[3]; areas[3] = t
            t = areas[1]; areas[1] = areas[2]; areas[2] = t
        ret = 0
        for pred, area in zip(preds, areas):
            ret = ret + pred * (area / tot_area).unsqueeze(-1)
        return ret

    def forward(self, inp1, coord1, cell1, inp2):
        return self.query_rgb(inp1, coord1, cell1, inp2)

def make_coord(shape, ranges=None, flatten=True):
    """ Make coordinates at grid centers.
    """
    coord_seqs = []
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs), dim=-1)
    if flatten:
        ret = ret.view(-1, ret.shape[-1])
    return ret
