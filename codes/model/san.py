import math
import torch
import numpy as np
from torch import nn
from model.basic_modules import *
from .DCN_v2.modules.modulated_deform_conv import *

### Multi-Scale Feature Fusion (MSFF) 
class MSFF(nn.Module):
    def __init__(self, channel, ksize):
        super(MSFF, self).__init__()

        self.channel = channel
        self.mlp = MLP(channel,  channel,  hidden_list=[channel,channel,channel])
        self.dcn = ModulatedDeformConvPack(channel,  channel,  kernel_size=(ksize, ksize), stride=1, padding=ksize//2, deformable_groups=1)
    
    def forward(self, ev_feat, im_feat):
        B,C,H,W = im_feat.shape
        coord = make_coord((H, W)).unsqueeze(0).expand(B,-1,-1).to(im_feat.device)
        cell = torch.ones_like(coord, device=coord.device)
        cell[:,:, 0] *= 2 / H
        cell[:,:, 1] *= 2 / W
        feat = self.mlp(ev_feat, coord, cell, im_feat) 
        feat = feat.view(B, H, W, self.channel).permute((0, 3, 1, 2))
        feat = self.dcn(feat.contiguous())

        return feat

### Scale-Aware Network (SAN) 
class San(nn.Module):
    def __init__(self, im_in_channel, out_channel, ev_in_channel, initializor='kaiming'):
        super(San, self).__init__()

        ## encoders and decoder
        self.im_enc = Encoder(im_in_channel)
        self.ev_enc = Encoder(ev_in_channel)
        self.dec = Decoder(out_channel)
        self.softplus = nn.Softplus()

        ## MSFF modules
        self.msff1 = MSFF(16, 5)
        self.msff2 = MSFF(32, 3)
        self.msff3 = MSFF(64, 3)
        self.msff4 = MSFF(128, 3)
        self.msff5 = MSFF(128, 3)
        
        ## init net
        init_weights(self.im_enc, initializor)
        init_weights(self.dec, initializor)
        init_weights(self.ev_enc, initializor)
        
    def forward(self, event, blur):
        
        blur = (blur - 0.5) / 0.5
        
        ev_feat1, ev_feat2, ev_feat3, ev_feat4, ev_feat5 = self.ev_enc(event)
        im_feat1, im_feat2, im_feat3, im_feat4, im_feat5 = self.im_enc(blur)
        
        L_feat1 = self.msff1(ev_feat1, im_feat1)
        L_feat2 = self.msff2(ev_feat2, im_feat2)
        L_feat3 = self.msff3(ev_feat3, im_feat3)
        L_feat4 = self.msff4(ev_feat4, im_feat4)
        L_feat5 = self.msff5(ev_feat5, im_feat5)
        
        SAN_E_out = self.dec(L_feat1, L_feat2, L_feat3, L_feat4, L_feat5)
        SAN_E_out = self.softplus(SAN_E_out)

        L = (blur + 1) / 2 * SAN_E_out
    
        return L, SAN_E_out

    def freeze(self):
        for p in self.parameters():
            p.requires_grad = False

        