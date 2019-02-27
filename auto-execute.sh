ori_path='/home/wdd/pytorch-vdsr-3d/test/testset/test1'
inter_path='/home/wdd/pytorch-vdsr-3d/test/x3interp/test1x3'
model_path='model_12layers_25input_3kernel_multi/model_epoch_31.pkl'
block_size=50
reconstrucion_size=400
count=(reconstruction_size/block_size)-1
for ((i=0;i<=3;i++))
do
for ((j=0;j<=3;j++))
do
for ((k=0;k<=3;k++))
do
python generate_subblocks.py --countd $i --counth $j --countw $k --interpath $inter_path --model $model_path
done
done
done
python restore_complete_3Dimage.py --oripath $ori_path --interpath $inter_path  
