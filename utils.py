import os
import os.path
import random
import numpy as np
import cv2
import torch
import torch.utils.data as data
import torch.nn as nn
import torch.nn.functional as F 
from torch.nn import init
from torch.autograd import Variable
import torch.optim as optim
import DIV2K_Dataset as DIV2K


def getPatch(inimg, tarimg, patch_size=96, scale=4):
    '''
    get patch of img
    '''
    # get height, width and channel of inimg
    (ih, iw, c) = inimg.shape
    (th, tw) = (scale * ih, scale * iw)
    tp = patch_size
    ip = patch_size // scale
    ix = random.randrange(0, iw - ip + 1)
    iy = random.randrange(0, ih - ip + 1)
    tx, ty = scale * ix, scale * iy
    inimg = inimg[iy:iy + ip, ix:ix + ip, :]
    tarimg = tarimg[ty:ty + tp, tx:tx + tp, :]
    
    return inimg, tarimg


def augment(inimg, tarimg):
    '''
    data augment for img
    '''
    if random.random() < 0.3:
        inimg = inimg[:, ::-1, :]
        tarimg = tarimg[:, ::-1, :]
    if random.random() < 0.3:
        inimg = inimg[::-1, :, :]
        tarimg = tarimg[::-1, :, :]
    return inimg, tarimg


def rgb_to_tensor(inimg, tarimg):
    '''
    transform rgb to tensor
    '''
    ts = (2, 0, 1)
    inimg = torch.Tensor(inimg.transpose(ts).astype(float)).mul_(1.0)
    tarimg = torch.Tensor(tarimg.transpose(ts).astype(float)).mul_(1.0)  
    return inimg, tarimg


# img to tensor
class ToTensor(object):
    def __call__(self,sample):
        inputs = sample['input']
        labels = sample['label']
        inputs = np.ascontiguousarray(np.transpose(inputs,(2,0,1)))
        labels = np.ascontiguousarray(np.transpose(labels,(2,0,1)))
        return {"input":torch.from_numpy(inputs).float()/255.0,
              "label":torch.from_numpy(labels).float()/255.0}
 


# Flippe the img 
class Flippe(object):
    def __call__(self,sample):
        '''
        sample:
            @'inputs':32 32 lr_img
            @'labels':128*128 hr_img 
        '''
        is_hor  = random.random()>0.5
        inputs = sample['input']
        labels = sample['label']
        #whether hor flip
        if is_hor:
            inputs = inputs[:,::-1,:]
            labels = labels[:,::-1,:]
        return {"input":inputs,"label":labels}

    
# random to rotate
class Rotation(object):
    def __call__(self,sample):
        is_rot = random.random()>0.5
        inputs = sample['input']
        labels = sample['label']
        if is_rot:
            inputs = np.transpose(inputs,(1,0,2))
            labels = np.transpose(labels,(1,0,2))
        return {"input":inputs,"label":labels}
    
    
def weights_init_kaiming(m):
    '''
    init weights
    '''
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        init.kaiming_normal(m.weight.data)
        

def set_lr(lr, decayType, decay, epoch, optimizer):
    '''
    set lr in each epoch
    decayType = 0: step by step
    decayType = 1: exp
    decayType = 2: inv
    '''
    if decayType == 0:
        epoch_iter = (epoch + 1) // decay
        nlr = lr / 2**epoch_iter
    elif decayType == 1:
        k = math.log(2) / decay
        nlr = lr * math.exp(-k * epoch)
    elif decayType == 2:
        k = 1 / decay
        nlr = lr / (1 + k * epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = nlr
    return nlr
    