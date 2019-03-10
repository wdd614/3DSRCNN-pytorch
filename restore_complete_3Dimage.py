#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 12 22:51:06 2018
Restroing the large scale image from sub-blocks which is 
reconstructed from the 3d-VDSR.
@author: wdd
@Copyright Notice:Please contact 1556905690@qq.com before you reprint this program

"""


import utils
import numpy as np
import argparse
import os

parser = argparse.ArgumentParser(description="restroing the large scale image")
parser.add_argument("--oripath", type=str, default='/home/wdd/pytorch-vdsr-3d/testset/test1',help="path to original image ")
parser.add_argument('--interpath',type=str,default='/home/wdd/pytorch-vdsr-3d/testset/x2interp/test3x2',help='path to interpolation image ')
parser.add_argument('--imagesize',type=str,default=400,help='size of reconstructed image')
parser.add_argument('--block_size',type=int,default=100,help='reconstrucion size per time')
parser.add_argument('--sub_name',type=str,default='11')
opt=parser.parse_args()
#dataset_interp=utils.read_imagecollection('/home/wdd/pytorch-vdsr-3d/testset/x2interp/test1x2')
num,h,w = (opt.imagesize,opt.imagesize,opt.imagesize)
batch_generate_size = 100
reconstruction_size = 400
reconstruction_output = np.zeros((num,h,w))
default_path = 'gen_results/3D_VDSR_'
if not os.path.exists(default_path):
    os.makedirs("gen_results/")

sub_name = opt.sub_name
#utils.read_imagecollection('/home/wdd/pytorch-vdsr-3d/3D_VDSR_001')
for count_d in range((num//batch_generate_size)):
    for count_h in range((h//batch_generate_size)):
        for count_w in range((w//batch_generate_size)):
            pixel_start_d=count_d*batch_generate_size
            pixel_end_d=(count_d+1)*batch_generate_size
            pixel_start_h=count_h*batch_generate_size
            pixel_end_h=(count_h+1)*batch_generate_size
            pixel_start_w=count_w*batch_generate_size
            pixel_end_w=(count_w+1)*batch_generate_size
            sub_block=utils.read_imagecollection(file_path=default_path+str(count_d)+str(count_h)+str(count_w))
            #print ('current path is:'+'3D_VDSR_'+str(count_d)+str(count_h)+str(count_w),'shape of sub_block is:',sub_block.shape)
            reconstruction_output[pixel_start_d:pixel_end_d,pixel_start_h:pixel_end_h,pixel_start_w:pixel_end_w]=sub_block

dataset_ori = utils.read_imagecollection(opt.oripath)
dataset_interp = utils.read_imagecollection(opt.interpath)
#print ('======>Read original image from : ',opt.oripath,' Read low resolution images from : ',opt.interpath)
#print ('PSNR of interp:',utils.PSNR(dataset_interp[:reconstruction_size,:reconstruction_size,:reconstruction_size],dataset_ori[:reconstruction_size,:reconstruction_size,:reconstruction_size]))
#print ('PSNR of reconstructor:',utils.PSNR(reconstruction_output,dataset_ori[:reconstruction_size,:reconstruction_size,:reconstruction_size]))
print (utils.PSNR(reconstruction_output,dataset_ori[:reconstruction_size,:reconstruction_size,:reconstruction_size]))
utils.generate_2Dimage(save_mode='3DVDSR_'+sub_name+'/',array_like=reconstruction_output) 

