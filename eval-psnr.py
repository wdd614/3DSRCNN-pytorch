#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  8 11:01:47 2018

@author: wdd
"""

import argparse
import torch
from torch.autograd import Variable
from scipy.ndimage import imread
from PIL import Image
import numpy as np
import time, math
import matplotlib.pyplot as plt
import os
from skimage import data,io
import scipy.ndimage

def PSNR(pred, gt, shave_border=0):
#    height, width = pred.shape[:2]
#    pred = pred[shave_border:height - shave_border, shave_border:width - shave_border]
#    gt = gt[shave_border:height - shave_border, shave_border:width - shave_border]
    imdff = pred - gt
    rmse = math.sqrt(np.mean(imdff ** 2))
    if rmse == 0:
        return 100
    return 20 * math.log10(255.0 / rmse)


path_ori=os.path.join(os.getcwd(),'/home/wdd/桌面/testset/ori1/*.bmp')
path_interp=os.path.join(os.getcwd(),'/home/wdd/桌面/testset/interp/*.bmp')
# print (path)
imgs_ori=io.imread_collection(path_ori)#使用读取一组图片
imgs_interp=io.imread_collection(path_interp)
dataset_ori=[]
dataset_interp=[]
for img in imgs_ori:
#     img=img.reshape(1,400,400)
    dataset_ori.append(img)
for img in imgs_interp:
    dataset_interp.append(img)
dataset_ori=np.asarray(dataset_ori).astype(np.float)
dataset_interp=np.asarray(dataset_interp).astype(np.float)

#     print (img.shape,type(img))#ndarray尺寸已经改好了(1,400,400)，提升一个维度
scale=2
zoom=scipy.ndimage.interpolation.zoom(dataset_ori,1./scale)
zoom=scipy.ndimage.interpolation.zoom(zoom,scale)#shape(396,396,396)
print ('zoom psnr:',PSNR(dataset_ori,zoom))
print ('interp psnr:',PSNR(dataset_ori,dataset_interp))
