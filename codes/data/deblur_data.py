import sys # remove the path of ROS
if '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import os 
import cv2
import torch
import numpy as np
from . import utils
from os.path import join as jn
import torch.utils.data as data
from torch.nn import functional as F

class DeblurData(data.Dataset):
    def __init__(self, root_path, num_bins=16, roi_size=(128,128), scale_factor=4, has_gt=True, train=True, predict_ts=None):
        # Set all input args as attributes
        self.__dict__.update(locals())

        # basic params
        self.check_files(root_path)
        self.count = 0
        self.train = train
        self.num_bins = num_bins
        self.scale_factor = scale_factor

        # training params
        self.ev_roi_size = roi_size
        self.im_roi_size = (roi_size[0] * scale_factor, roi_size[1] * scale_factor)

        # test params
        self.has_gt = has_gt

        # prediction params
        self.predict_ts = predict_ts

    def get_filename(self, path, suffix):
        # get filenames with specific suffix from path
        namelist=[]
        filelist = os.listdir(path)
        for i in filelist:
            if os.path.splitext(i)[1] == suffix:
                namelist.append(i)
        namelist.sort()
        return namelist

    def check_files(self, root_path):
        # check file structure 
        middir = 'train' if self.train else 'test'
        self.data_path = jn(root_path, middir)
        self.data_list = self.get_filename(self.data_path, '.npz')

    def get_data(self, idx):
        # get data from npz files
        data = np.load(jn(self.data_path, self.data_list[idx]), allow_pickle=True)
        events = data['events'].item()

        blur1 = data['blur1'].astype(float)
        exp_start1 = data['exp_start1']
        exp_end1 = data['exp_end1']

        blur2 = data['blur2'].astype(float)
        exp_start2 = data['exp_start2']
        exp_end2 = data['exp_end2']

        if self.train:
            return events, blur1, exp_start1, exp_end1, blur2, exp_start2, exp_end2
        else:
            prefix = os.path.splitext(self.data_list[idx])[0] # prefix of the filename
            if self.has_gt:
                sharp_imgs = data['sharp_imgs']
                timestamps = data['sharp_timestamps']
                return events, blur1, exp_start1, exp_end1, blur2, exp_start2, exp_end2, timestamps, sharp_imgs, prefix
            else:
                return events, blur1, exp_start1, exp_end1, blur2, exp_start2, exp_end2, prefix

    def get_training_data(self, idx):
        # load data for training
        events, blur1, exp_start1, exp_end1, blur2, exp_start2, exp_end2 = self.get_data(idx)

        # generate large blur
        large_blur = ((blur1.astype(float) + blur2.astype(float)) / 2).clip(0,255)
        large_blur_span = (exp_start1, exp_end2)

        # select blur
        blur_idx = np.random.randint(0, 2)
        if blur_idx: # use blur1
            blur = blur1
            blur_span = (exp_start1, exp_end1)
            sharp_ts = np.random.randint(exp_start1, exp_end1)
        else: # use blur2
            blur = blur2
            blur_span = (exp_start2, exp_end2)
            sharp_ts = np.random.randint(exp_start2, exp_end2)
        sharp_span = (sharp_ts, sharp_ts)

        # crop roi
        ev_size = (blur.shape[-2] // self.scale_factor, blur.shape[-1] // self.scale_factor)
        ev_roiTL = (np.random.randint(0, ev_size[0]-self.roi_size[0]+1), np.random.randint(0, ev_size[1]-self.roi_size[1]+1)) # top-left coordinate of event roi
        im_roiTL = (ev_roiTL[0] * self.scale_factor, ev_roiTL[1] * self.scale_factor) # top-left coordinate of image roi

        blur = blur[:,im_roiTL[0]:im_roiTL[0]+self.im_roi_size[0], im_roiTL[1]:im_roiTL[1]+self.im_roi_size[1]] 
        large_blur = large_blur[:,im_roiTL[0]:im_roiTL[0]+self.im_roi_size[0], im_roiTL[1]:im_roiTL[1]+self.im_roi_size[1]] 

        # generate events
        eger_blur2sharp = utils.gen_EGER(events, self.num_bins, sharp_span, blur_span, roiTL=ev_roiTL, roi_size=self.ev_roi_size)
        eger_largeblur2sharp = utils.gen_EGER(events, self.num_bins, sharp_span, large_blur_span, roiTL=ev_roiTL, roi_size=self.ev_roi_size)
        eger_largeblur2blur = utils.gen_EGER(events, self.num_bins, blur_span, large_blur_span, roiTL=ev_roiTL, roi_size=self.ev_roi_size)

        # numpy to tensor
        blur, large_blur = utils.np2Tensor(blur, large_blur, scale=1/255.)
        eger_blur2sharp, eger_largeblur2sharp, eger_largeblur2blur = utils.np2Tensor(eger_blur2sharp, eger_largeblur2sharp, eger_largeblur2blur, scale=1.)

        self.count += 1

        return eger_blur2sharp, eger_largeblur2sharp, eger_largeblur2blur, blur, large_blur

    def get_test_data(self, idx):

        # load data for test
        if self.has_gt:
            events, blur1, exp_start1, exp_end1, blur2, exp_start2, exp_end2, timestamps, sharp_imgs, prefix = self.get_data(idx)
        else:
            events, blur1, exp_start1, exp_end1, blur2, exp_start2, exp_end2, prefix = self.get_data(idx)

        eger_list = []
        blur_list = []
        gt_list = []

        assert not ((self.has_gt==False) and (self.predict_ts==None)), "Please enable has_gt or specify predict_ts"

        if self.predict_ts == None:
            # recover images at timestamps
            target_timestamps = timestamps
        else:
            # recover images based on predict_ts
            target_timestamps = []
            target_timestamps.append(exp_start1 + self.predict_ts * (exp_end1 - exp_start1))
            target_timestamps.append(exp_start2 + self.predict_ts * (exp_end2 - exp_start2))

        for i in range(len(target_timestamps)):
            # select input
            if target_timestamps[i] <= exp_end1:
                blur, exp_start, exp_end = blur1, exp_start1, exp_end1
            else:
                blur, exp_start, exp_end = blur2, exp_start2, exp_end2

            # generate eger
            sharp_span = (target_timestamps[i], target_timestamps[i])
            blur_span = (exp_start, exp_end)
            ev_size = (blur.shape[-2] // self.scale_factor, blur.shape[-1] // self.scale_factor)
            eger = utils.gen_EGER(events, self.num_bins, sharp_span, blur_span, roiTL=(0, 0), roi_size=ev_size)

            # collect data   
            eger_list.append(eger)
            blur_list.append(blur)
            if self.has_gt:
                gt_list.append(sharp_imgs[i,...])

        # numpy to tensor
        blur_list, gt_list, eger_list = np.array(blur_list), np.array(gt_list), np.array(eger_list)
        eger_list = utils.np2Tensor(eger_list, scale=1.)[0]
        blur_list = utils.np2Tensor(blur_list, scale=1/255.)[0]

        self.count += 1

        if self.has_gt:
            gt_list = utils.np2Tensor(gt_list, scale=1/255.)[0]
            return eger_list, blur_list, gt_list, prefix
        else:
            return eger_list, blur_list, prefix

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):

        if self.train:
            eger_blur2sharp, eger_largeblur2sharp, eger_largeblur2blur, blur, large_blur = self.get_training_data(idx)
            return eger_blur2sharp, eger_largeblur2sharp, eger_largeblur2blur, blur, large_blur
        else:
            if self.has_gt:
                eger, blur, gt, prefix = self.get_test_data(idx)
                return eger, blur, gt, prefix
            else:
                eger, blur, prefix = self.get_test_data(idx)
                return eger, blur, prefix
