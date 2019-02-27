#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13 09:19:20 2018
Used to calculate the parameters and saving images 
@author: wdd
"""

import numpy as np
import matplotlib.pyplot as plt
import math
import os
from skimage import io
import cv2


def PSNR(pred, gt):#this function(tested) can be used in 2D or 3D
    imdff = pred - gt
    rmse = math.sqrt(np.mean(imdff ** 2))
    if rmse == 0:
        return 100
    return 20 * math.log10(255.0 / rmse)
    

def generate_2Dimage(array_like,save_mode='3D_VDSR_/',image_format='bmp'):
    if not isinstance(array_like,np.ndarray):
        array_like=np.asarray(array_like)
#    shape=array_like.shape()
    if not os.path.exists(save_mode):
        os.mkdir(save_mode)
    for count,every_image in enumerate(array_like):
        cv2.imwrite(save_mode+str(count+1)+'.'+image_format,every_image)
    #print ('Save image to path:',save_mode)
    #print ('Successfully save '+str(count+1)+' '+image_format+'image!')

def display_2Dimage(image_array,mode='gray'):
    plt.imshow(image_array,cmap=mode)
    plt.show()

        
def read_imagecollection(file_path,image_format='bmp'):#
    imageset_path=os.path.join(os.getcwd(),file_path)
    if not os.path.exists(imageset_path):
        raise IOError
    imgs=io.imread_collection(imageset_path+'/*.'+image_format)
    imgs_arrayset=[]
    for img in imgs:
        imgs_arrayset.append(img)
    imgs_arrayset=np.asarray(imgs_arrayset).astype(np.float)
    #print ('Shape of imageset is:',imgs_arrayset.shape)
    return imgs_arrayset

def pre_crop(image_array,reconstrution_size=400):
    real_shape=image_array.shape
    temp=np.zeros((reconstrution_size,reconstrution_size,reconstrution_size))
    temp[:real_shape[0],:real_shape[1],:real_shape[2]]=image_array
    return temp
        
