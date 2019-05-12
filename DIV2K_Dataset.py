import os
import os.path as path
import random
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
import utils
from utils import *
from torchvision import transforms as T 
import visdom
import torchvision





'''
 the class for DIV2K dataset
'''

class DIV2K_Dataset(Dataset):
    def __init__(self, root='./data', scale=1, train=True, transform=None):
        '''
        init dataset with root dir
        scale: scale of image to be patched
        self.dirIn: input dir
        self.dirTar: target dir
        self.fileList: list of file
        '''
        self.root = root
        self.scale = scale
        self.train = train
        
        dirLR = 'LR/DIV2K_valid_LR_x8'
        dirHR = 'HR/DIV2K_valid_HR'
        
        if self.train:
            dirLR = 'LR/DIV2K_train_LR_x8'
            dirHR = 'HR/DIV2K_train_HR'
    
        self.transform = transform
        self.dirIn = path.join(self.root, dirLR)
        self.dirTar = path.join(self.root, dirHR)

        self.fileList = os.listdir(self.dirTar)
    
    def __getitem__(self, idx):
        '''
        get item by idx
        '''
        inName, tarName = self.getFileName(idx)
        inImg = cv2.imread(inName)
        tarImg = cv2.imread(tarName)
        
        if self.scale != 1:
            '''see if need to patch'''
            inImg, tarImg = utils.getPatch(inImg, tarImg, 96, self.scale)
        # data augment
        # inImg, tarImg = augment(inImg, tarImg)
        
        sample = {'input': inImg, 'label': tarImg}
        # RGB to Tensor
        # inTensor, tarTensor = rgb_to_tensor(inImg, tarImg)
        
        if self.transform is not None:
            sample = self.transform(sample)
        
        return sample
        
        
    
    def __len__(self):
        return len(self.fileList)
    
    # get file name
    def getFileName(self, idx):
        '''
        get input name and target name by idx
        '''
        name = self.fileList[idx]
        tarName = path.join(self.dirTar, name)
        name = name[0:-4] + 'x8' + '.png' # ?
        inName = path.join(self.dirIn, name)
        return inName, tarName
