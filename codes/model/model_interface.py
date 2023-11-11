import sys # remove the path of ROS
if '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import os
import cv2
import inspect
import torch
import numpy as np
import importlib
from torch import nn
from os.path import join as jn
from torch.nn import functional as F
from collections import OrderedDict
import torch.optim.lr_scheduler as lrs
from warmup_scheduler import GradualWarmupScheduler

import pytorch_lightning as pl
from .metrics import compare_psnr, compare_ssim

class MInterface(pl.LightningModule):
    def __init__(self, model_name, loss, lr, **kargs):
        super().__init__()
        self.save_hyperparameters(ignore=["callbacks"])
        self.configure_loss()
        self.model = self.load_model()
        self.teacher_model = self.load_model()

        if kargs['train']:
            # training params
            self.loss_weight = kargs['loss_weight'] # [loss_BC, loss_SC, loss_TG, loss_SG]
            self.training_stage = kargs['training_stage']
            assert self.training_stage in [1, 2]
            
            # load teacher model during second stage training
            if self.training_stage == 2:
                state_dict = self.load_state_dict_from_checkpoint(kargs['ckpt_path'])
                self.model.load_state_dict(state_dict)
                # load teacher model
                self.teacher_model.load_state_dict(state_dict)
                self.teacher_model.freeze()
                self.teacher_model.eval()
        else:
            # test params
            self.save_path = kargs['save_path']
            self.has_gt = kargs['has_gt']

    def forward(self, event, blur):
        return self.model(event, blur)

    def training_step(self, batch, batch_idx):
        eger_blur2sharp, eger_largeblur2sharp, eger_largeblur2blur, blur, large_blur = batch
        B, ev_C, ev_H, ev_W = eger_blur2sharp.shape
        im_C, im_H, im_W = blur.shape[-3:]
        assert im_H / ev_H == im_W / ev_W

        # assemble inputs
        egers = torch.cat((eger_blur2sharp[:,None,...],  eger_largeblur2blur[:,None,...], eger_largeblur2sharp[:,None,...]), 1).reshape((B*3,ev_C,ev_H,ev_W)) # B*3,T*4,H,W
        blurs = torch.cat((blur[:,None,...], large_blur[:,None,...], large_blur[:,None,...]), 1).reshape((B*3,im_C,im_H,im_W)) 

        # process by network
        L, SAN_E = self(egers, blurs)
        L = L.clamp(0,1).reshape((B,3,im_C,im_H,im_W))
        SAN_E = SAN_E.reshape((B,3,im_C,im_H,im_W))

        ## first stage training
        if self.training_stage == 1:
            # compute loss 
            loss_BC, loss_SC = self.compute_loss(L, SAN_E, blur)
            loss = self.loss_weight[0] * loss_BC + self.loss_weight[1] * loss_SC

        ## second stage training
        else:
            # generate pseudo-ground-truth by teacher model
            L_teacher, _ = self.teacher_model(eger_blur2sharp, blur)
            L_teacher = L_teacher.clamp(0,1)

            # generate down-sampled results
            rand_scale = ((np.random.randint(ev_H, im_H+1) // 32) * 32) / im_H
            lr_large_blur = self.image_downsampling(large_blur, rand_scale)
            lr_L_lb2s, _ = self(eger_largeblur2sharp, lr_large_blur)
            lr_L_lb2s = lr_L_lb2s.clamp(0,1)
            
            # generate down-sampled pseudo-ground-truth
            lr_L_teacher = self.image_downsampling(L_teacher, rand_scale)

            # compute loss  
            loss_BC, loss_SC, loss_TG, loss_SG = self.compute_loss(L, SAN_E, blur, lr_L_lb2s, L_teacher, lr_L_teacher)
            loss = self.loss_weight[0] * loss_BC + self.loss_weight[1] * loss_SC +\
                 self.loss_weight[2] * loss_TG + self.loss_weight[3] * loss_SG

        self.log('loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        if self.has_gt:
            eger, blur, gt, prefix = batch
        else:
            eger, blur, prefix = batch 

        # process by network
        B, N = eger.shape[:2]
        for n in range(N):
            L, _ = self(eger[:,n,...], blur[:,n,...])
            L = L.clamp(0,1)

            # save results
            if self.save_path != None:
                for b in range(B):
                    
                    # save reconstruction
                    result = (L[b,...].permute(1,2,0).squeeze().cpu().detach().numpy() * 255).astype(np.uint8)
                    save_name = prefix[b] + '_%03d.png' %(n)
                    cv2.imwrite(jn(self.result_root, save_name), result)

                    # save blurry inputs
                    blurry_img = (blur[b,n,...].permute(1,2,0).squeeze().cpu().detach().numpy() * 255).astype(np.uint8)
                    cv2.imwrite(jn(self.blur_root, save_name), blurry_img)

                    # save gt and compute metrics
                    if self.has_gt:
                        groundtruth = (gt[b,n,...].permute(1,2,0).squeeze().cpu().detach().numpy() * 255).astype(np.uint8)
                        cv2.imwrite(jn(self.gt_root, save_name), groundtruth) 

                        # compute metrics
                        psnr, ssim = compare_psnr(groundtruth, result), compare_ssim(groundtruth, result)
                        self.log('psnr', psnr, on_step=True, on_epoch=True, prog_bar=True)
                        self.log('ssim', ssim, on_step=True, on_epoch=True, prog_bar=True)

    def on_test_epoch_start(self):
        # ensure directories
        if self.save_path != None:
            self.result_root = jn(self.save_path, 'Result')
            if not os.path.exists(self.result_root):
                os.makedirs(self.result_root) 
            self.blur_root = jn(self.save_path, 'Blur')
            if not os.path.exists(self.blur_root):
                os.makedirs(self.blur_root) 

            if self.has_gt:
                self.gt_root = jn(self.save_path, 'GroundTruth')
                if not os.path.exists(self.gt_root):
                    os.makedirs(self.gt_root) 
    
    def compute_loss(self, L, SAN_E, blur, lr_L_lb2s=None, L_teacher=None, lr_L_teacher=None):
        # seperate results
        L_b2s, L_lb2b, L_lb2s = L[:,0,...], L[:,1,...], L[:,2,...]
        SAN_E_b2s, SAN_E_lb2b, SAN_E_lb2s = SAN_E[:,0,...], SAN_E[:,1,...], SAN_E[:,2,...]
        
        # compute first-stage losses
        loss_BC = self.loss_function(L_lb2b, blur)
        loss_SC = self.loss_function(SAN_E_lb2s / SAN_E_b2s, SAN_E_lb2b)

        if self.training_stage == 1:
            return loss_BC, loss_SC
        else:
            # compute second-stage losses
            loss_TG = self.loss_function(L_teacher, L_lb2s)
            loss_SG = self.loss_function(lr_L_teacher, lr_L_lb2s) # may need to be modified
            return loss_BC, loss_SC, loss_TG, loss_SG

    def image_downsampling(self, img, scale_factor, mode='bicubic'):
        img = F.interpolate(img, scale_factor=scale_factor, mode=mode).clamp(0,1)
        return img

    def load_state_dict_from_checkpoint(self, checkpoint_path):
        state_dict = torch.load(checkpoint_path)['state_dict']
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if k[:6] == 'model.': # find keys starting with `model.`
                name = k[6:] # remove `model.`
                new_state_dict[name] = v
        return new_state_dict

    def configure_optimizers(self):
        if hasattr(self.hparams, 'weight_decay'):
            weight_decay = self.hparams.weight_decay
        else:
            weight_decay = 0
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.hparams.lr, weight_decay=weight_decay)

        if self.hparams.lr_scheduler is None:
            return optimizer
        else:
            if self.hparams.lr_scheduler == 'step':
                scheduler = lrs.StepLR(optimizer,
                                       step_size=self.hparams.lr_decay_steps,
                                       gamma=self.hparams.lr_decay_rate)
            elif self.hparams.lr_scheduler == 'cosine':
                scheduler = lrs.CosineAnnealingLR(optimizer,
                                                  T_max=self.hparams.lr_decay_steps,
                                                  eta_min=self.hparams.lr_decay_min_lr)
            else:
                raise ValueError('Invalid lr_scheduler type!')

            if self.hparams.warmup_epoch > 0:
                scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=self.hparams.warmup_epoch, after_scheduler=scheduler)
                return [optimizer], [scheduler_warmup]
            else:
                return [optimizer], [scheduler]

    def configure_loss(self):
        loss = self.hparams.loss.lower()
        if loss == 'mse':
            self.loss_function = F.mse_loss
        elif loss == 'l1':
            self.loss_function = F.l1_loss
        else:
            raise ValueError("Invalid Loss Type!")

    def load_model(self):
        name = self.hparams.model_name
        # Change the `snake_case.py` file name to `CamelCase` class name.
        # Please always name your model file name as `snake_case.py` and
        # class name corresponding `CamelCase`.
        camel_name = ''.join([i.capitalize() for i in name.split('_')])
        try:
            Model = getattr(importlib.import_module(
                '.'+name, package=__package__), camel_name)
        except:
            raise ValueError(
                f'Invalid Module File Name or Invalid Class Name {name}.{camel_name}!')

        return self.instancialize(Model)

    def instancialize(self, Model, **other_args):
        """ Instancialize a model using the corresponding parameters
            from self.hparams dictionary. You can also input any args
            to overwrite the corresponding value in self.hparams.
        """
        class_args = inspect.getargspec(Model.__init__).args[1:]
        inkeys = self.hparams.keys()
        args1 = {}
        for arg in class_args:
            if arg in inkeys:
                args1[arg] = getattr(self.hparams, arg)
        args1.update(other_args)
        return Model(**args1)

