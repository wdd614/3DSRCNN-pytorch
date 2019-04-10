# 3DSRCNN-pytorch  
## How to make trainig dataset for?  
This part is coded by matlab.  
## Where to find Training Data?
The traing data is generated through 5 CT sets, summed up to  2000 single 2D images totally. 
So this is so big training data library. We generated 32.5G of data, it is inconvenient to upload to Github.
If your would like to trainig network, please concat me with 1556905690@qq.com or issue in this Repository.
## How to train a 3D super resolution Network?  
It is easy to train our network, please
```
python main.py 

--batchSize", type=int, default=64, help="Training batch size"
--nEpochs", type=int, default=100, help="Number of epochs to train for"
-lr", type=float, default=0.1, help="Learning Rate. Default=0.1"
--step", type=int, default=10, help="Sets the learning rate to the initial LR decayed by momentum every n epochs, Default: n=10"
--cuda", type=int,default=1, help="Use cuda?"
--resume", default="", type=str, help="Path to checkpoint (default: none)"
--start-epoch", default=1, type=int, help="Manual epoch number (useful on restarts)"
--clip", type=float, default=0.4, help="Clipping Gradients. Default=0.4")
--threads", type=int, default=1, help="Number of threads for data loader to use, Default: 1"
--momentum", default=0.9, type=float, help="Momentum, Default: 0.9"
--weight-decay", "--wd", default=1e-4, type=float, help="Weight decay, Default: 1e-4"
--pretrained', default='', type=str, help='path to pretrained model (default: none)'
--train_path',type=str,default="train_data/3dtrain25all.h5",help='Path to train dataset'
--memo', default= 'L_', type=str, help='prefix of logger '
```
## how to modify our network 
`3dsrcnn.py` This program is a structure for building a network.  ``
## Automatically Reconstruct and calculate PSNR  
Because our validation dataset is very complex which contains 5 sets, each set contains x2, x3, x4 scale.
What's more, we need to calculate PSNR of every Epoch. We provided automatically executing script.
  
Note that you must specify path to low resoluiton images(inter_path)
original HR path (ori_path)
`bash auto-execute.sh`
if you want compare multi-scale or model of specified epoch, you could run 
```
bash auto-excute=ulti_testset.sh
```
For example, the generated log is follwing:
```

x2, x3, x4 scale PSNR of model/0310-2114_model/model_epoch_35.pkl 
-----------------
2019/3/  23 Tue 07:21:31 CST
33.37136136045343
30.771517328760474
35.299094237934455
39.6395555291764
45.80099485177503
------------------
2019/3/ 21 Tue 08:04:13 CST
29.78923091882109
26.636256131118948
32.5371038561331
37.378719766068414
44.580486528475156
-----------------
2019/ 3 21 Tue 08:46:30 CST
26.629647924465274
23.407602537288376
31.092819773019947
35.71189104481404
42.932984128665936
-----------------
```
the acerage PSNR will be generated and  store in output.txt
**Note that, If you want to use `auto-execute.sh`, you must promise 
established file-tree format!**
```
ori_path='30x30_2'#HR_path
inter_path='30x30_2'#LR_path(under feeding into network)
model_path='model_12layers_25input_3kernel_multi/model_epoch_20.pkl'#specify model path
saved_prefix='30x30_'#prefix file path  to save generated images
format='bmp'#reading images format
```
