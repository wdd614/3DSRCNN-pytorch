# 3DSRCNN-pytorch  
## How to make trainig dataset for?  
This part is coded by matlab.  

## How to train a 3D super resolution Network?  

```
python main.py
```
## how to modify our network 
`3dsrcnn.py` is the main network structrue.  ``
## Automatically Reconstruct and calculate PSNR  
We provided script which could automatically generate data and calculate PSNR.
Note that you must specify path to low resoluiton images(inter_path)
original HR path (ori_path)
`bash auto-execute.sh`
if you want compare multi-scale or model of specified epoch, you could run 
````
bash auto-excute=ulti_testset.sh
```

the acerage PSNR will be generated and  store by different salce 