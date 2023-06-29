import os
import torch
import numpy as np
import torch.nn.functional as F

def np2Tensor(*args, scale=1.):
    # convert numpy to tensor
    def _np2Tensor(img):
        np_transpose = np.ascontiguousarray(img)
        tensor = torch.from_numpy(np_transpose).float()
        tensor.mul_(scale)
        return tensor

    return [_np2Tensor(a) for a in args]

def filter_events_by_key(key,x1,x2,x3, value1, value2): 
    # filter events based on key dimension (start inclusive and end exclusive)
    min_value = min(value1, value2)
    max_value = max(value1, value2)
    mask = (value1 <= key) * (key < max_value)
    
    new_x1 = x1[mask]
    new_x2 = x2[mask]
    new_x3 = x3[mask]
    new_key = key[mask]

    return new_key,new_x1,new_x2,new_x3

def event2frame(E, x, y, t, p, start, end, interval, exp_start):
    # convert events to frame-like representation
    num_bins, C, H, W = E.shape
    t,x,y,p = filter_events_by_key(t,x,y,p, start, end) 
    t -= int(exp_start) # shift minima
    idx = np.floor(t / interval).astype(int)
    idx[idx==num_bins] -= 1

    E = E.ravel()
    np.add.at(E, x + y*W + p*W*H + idx*W*H*2, 1)
    E = np.reshape(E, (num_bins, C, H, W))
    return E

def gen_EGER(event, num_bins, target_span, blur_span, roiTL=(0,0), roi_size=(256,256)):
    # generate exposure guided event representation (EGER)
    x = event['x'].astype(int)
    y = event['y'].astype(int)
    p = event['p'].astype(int)
    t = event['t'].astype(np.int64)
    
    H, W = roi_size
    blur_start, blur_end = blur_span
    target_start, target_end = target_span
    
    # filter events by blur span
    t,x,y,p = filter_events_by_key(t,x,y,p, blur_start, blur_end) # filter events by t dim
    x,y,p,t = filter_events_by_key(x,y,p,t,roiTL[1], roiTL[1]+roi_size[1]) # filter events by x dim
    y,x,p,t = filter_events_by_key(y,x,p,t,roiTL[0], roiTL[0]+roi_size[0]) # filter events by y dim
    x -= roiTL[1] # shift minima to zero
    y -= roiTL[0] # shift minima to zero
    p[p==-1] = 0 # pos in channel 1, neg in channel 0
    interval = (blur_end - blur_start) / num_bins

    # generate E1
    E1 = np.zeros((num_bins,2,H,W))
    E1 = event2frame(E1, x, y, t, p, blur_start, target_start, interval, blur_start)

    # generate E2
    E2 = np.zeros((num_bins,2,H,W))
    E2 = event2frame(E2, x, y, t, p, target_start, target_end, interval, blur_start)

    # generate E3
    E3 = np.zeros((num_bins,2,H,W))
    E3 = event2frame(E3, x, y, t, p, target_end, blur_end, interval, blur_start)
    
    eger = np.concatenate((E1, E2, E3), 1) 
    eger = eger.reshape((num_bins * 2 * 3, H,W))
    
    return eger
