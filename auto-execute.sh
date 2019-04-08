#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=2

ori_path='30x30_2'

inter_path='30x30_2'
model_path='model_12layers_25input_3kernel_multi/model_epoch_20.pkl'
saved_prefix='30x30_'
format='bmp'
block_size=100
reconstrucion_size=400
count=(reconstruction_size/block_size)-1
date 
for ((i=0;i<=3;i++))
do
for ((j=0;j<=3;j++))
do
for ((k=0;k<=3;k++))
do
python generate_subblocks.py --countd $i --counth $j --countw $k --interpath $inter_path --model $model_path  --format $format
done
done
done
python restore_complete_3Dimage.py --oripath $ori_path --interpath $inter_path  --memo $saved_prefix 
date 
