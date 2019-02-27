
# -*- coding: utf-8 -*-

import torch
from torch.autograd import Variable
import numpy as np
import argparse
import utils

parser = argparse.ArgumentParser(description="PyTorch VDSR")
parser.add_argument("--countd", type=int, default=0, help="d_size")
parser.add_argument("--counth", type=int, default=0, help="h_size")
parser.add_argument("--countw", type=int, default=0, help="w_size")
parser.add_argument('--interpath',type=str,default='/home/wdd/pytorch-vdsr-3d/testset/test1',help='path to interpolation images' )
parser.add_argument('--crop',type=int,default=0,help='pre-processing')
parser.add_argument('--block_size',type=int,default=100,help='reconstrucion size per time')
parser.add_argument('--model',type=str,default="model/model_epoch_1.pkl",help='path to trained model')
parser.add_argument('--cuda',type=int,default=1)
opt=parser.parse_args()


        
#opt = parser.parse_args()
CUDA_ENABLE = opt.cuda
PRE_CROP=opt.crop
#if CUDA_ENABLE and not torch.cuda.is_available():
#    raise Exception("No GPU found, please run without --cuda")

model_path=opt.model#current filepath 
model = torch.load(model_path)['model']
#params=model.state_dict()
#print (params)

if CUDA_ENABLE:
    model = model.cuda()
    #print ('Using GPU acceleration!')
else :
    model=model.cpu()
    print ('Using CPU to compute')
    
image_path=opt.interpath
if PRE_CROP :
    dataset_interp=utils.read_imagecollection(image_path)
    dataset_interp=utils.pre_crop(dataset_interp)
else:
    dataset_interp=utils.read_imagecollection(image_path)
#print('===>Load input low resolution image from : ',image_path)
dataset_interp=dataset_interp/255#normlize the gray rank to 0-1
dataset_interp=dataset_interp[:400,:400,:400]

num,h,w=dataset_interp.shape
batch_generate_size=100
reconstruction_output=np.zeros((num,h,w))
count_d,count_h,count_w=opt.countd,opt.counth,opt.countw

pixel_start_d=count_d*batch_generate_size
pixel_end_d=(count_d+1)*batch_generate_size
pixel_start_h=count_h*batch_generate_size
pixel_end_h=(count_h+1)*batch_generate_size
pixel_start_w=count_w*batch_generate_size
pixel_end_w=(count_w+1)*batch_generate_size
testdata=dataset_interp[pixel_start_d:pixel_end_d,pixel_start_h:pixel_end_h,pixel_start_w:pixel_end_w]
#print ('input data from interplation:',testdata.shape)
testdata=testdata.reshape(1,1,batch_generate_size,batch_generate_size,batch_generate_size)
#            testdata=torch.cuda.FloatTensor(testdata)
testdata=torch.Tensor(testdata)
if  CUDA_ENABLE:
    testdata_variable=Variable(testdata).cuda()
    testdata_output=model(testdata_variable)
    output=testdata_output.data.cpu().numpy().squeeze()
else :
    testdata_variable=Variable(testdata)
    testdata_output=model(testdata_variable)
    output=testdata_output.data.numpy().squeeze()
output=output*255#restore to the gray rank0-255
reconstruction_output[pixel_start_d:pixel_end_d,pixel_start_h:pixel_end_h,pixel_start_w:pixel_end_w]=output#
#del testdata_variable

#dataset_ori=read_imagecollection('/home/wdd/桌面/testset/ori1/*.bmp')
dataset_interp=dataset_interp*255
#print ('PSNR of interp:',PSNR(dataset_interp,dataset_ori[:400,:400,:400]))
#print ('PSNR of reconstructor:',PSNR(reconstruction_output,dataset_ori[:400,:400,:400]))
utils.generate_2Dimage(output,save_mode='3D_VDSR_'+str(count_d)+str(count_h)+str(count_w)+'/')
