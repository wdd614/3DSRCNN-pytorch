import torch
from torch.autograd import Variable
from scipy.ndimage import imread
import numpy as np
import time, math
import matplotlib.pyplot as plt
import time, math
import matplotlib.pyplot as plt
import os
import torch.nn as nn
from skimage import data,io
import cv2

torch.backends.cudnn.enabled = True


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
    if not os.path.exists('3D_VDSR_/'):
        os.mkdir(save_mode)
    for count,every_image in enumerate(array_like):
        cv2.imwrite(save_mode+str(count+1)+'.'+image_format,every_image)
    print ('Successfully save'+str(count)+image_format+'image!')

def display_2Dimage(image_array,mode='gray'):
    plt.imshow(image_array,cmap=mode)
    plt.show()

        
#opt = parser.parse_args()
CUDA_ENABLE = 0
#if CUDA_ENABLE and not torch.cuda.is_available():
#    raise Exception("No GPU found, please run without --cuda")

model_path="model/model_epoch_10.pth"#current filepath 
model = torch.load(model_path)['model']
model=model.cpu()
params=model.state_dict()
#for item in params:
#    print (item)
print(params['module.residual_layer.0.conv.weight'])

if CUDA_ENABLE:
    model = model.cuda()
    print ('Using GPU acceleration!')
else :
    model=model.cpu()
    
    print ('Using CPU to compute')
    

def read_imagecollection(file_path):
    imageset_path=os.path.join(os.getcwd(),file_path)
    imgs=io.imread_collection(imageset_path)
    imgs_arrayset=[]
    for img in imgs:
        imgs_arrayset.append(img)
    imgs_arrayset=np.asarray(imgs_arrayset).astype(np.float)
    print ('Shape of imageset is:',imgs_arrayset.shape)
    return imgs_arrayset
    

dataset_interp=read_imagecollection('/home/wdd/桌面/testset/interp/*.bmp')
dataset_interp=dataset_interp/255#normlize the gray rank to 0-1
dataset_interp=dataset_interp[:400,:400,:400]


num,h,w=dataset_interp.shape
batch_generate_size=100
reconstruction_output=np.zeros((num,h,w))
for count_d in range((num//batch_generate_size)+1):
    for count_h in range((h//batch_generate_size)+1):
        for count_w in range((w//batch_generate_size)+1):
            pixel_start_d=count_d*batch_generate_size
            pixel_end_d=(count_d+1)*batch_generate_size
            pixel_start_h=count_h*batch_generate_size
            pixel_end_h=(count_h+1)*batch_generate_size
            pixel_start_w=count_w*batch_generate_size
            pixel_end_w=(count_w+1)*batch_generate_size
            testdata=dataset_interp[pixel_start_d:pixel_end_d,pixel_start_h:pixel_end_h,pixel_start_w:pixel_end_w]
            print ('input data from interplation:',testdata.shape)
            testdata=testdata.reshape(1,1,batch_generate_size,batch_generate_size,batch_generate_size)
#            testdata=torch.cuda.FloatTensor(testdata)
            testdata=torch.Tensor(testdata)
            print (testdata)
            if  CUDA_ENABLE:
                testdata_variable=Variable(testdata).cuda()
                testdata_output=model(testdata_variable)
                output=testdata_output.data.cpu().numpy().squeeze()
                print ('Using GPU to accelerate....')
            else :
                testdata_variable=Variable(testdata).cpu()
                print (type(testdata_variable))
                testdata_output=model(testdata_variable)
                output=testdata_output.data.numpy().squeeze()
                print ('Using cpu to accelerate....')
            torch._C._cuda_emptyCache()
            output=output*255#restore to the gray rank0-255
            reconstruction_output[pixel_start_d:pixel_end_d,pixel_start_h:pixel_end_h,pixel_start_w:pixel_end_w]=output#
            del testdata_variable
            
dataset_ori=read_imagecollection('/home/wdd/桌面/testset/ori1/*.bmp')
dataset_interp=dataset_interp*255
print ('PSNR of interp:',PSNR(dataset_interp,dataset_ori[:400,:400,:400]))
print ('PSNR of reconstructor:',PSNR(reconstruction_output,dataset_ori[:400,:400,:400]))
generate_2Dimage(reconstruction_output)

