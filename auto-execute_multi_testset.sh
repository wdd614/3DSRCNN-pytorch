default_path='/home/wdd/pytorch-vdsr-3d/test/testset/test'
x='x'
interp='interp/'
path='/home/wdd/pytorch-vdsr-3d/test/'
t='test'
mp='model_12layers_25input_3kernel_multi/model_epoch_'
pkl='.pkl'
for ((model=35;model<=35;model++))
do
model_path=$mp$model$pkl
echo "===================>model start<================ " >>output_log.txt
date >>output_log.txt
echo $model_path >>output_log.txt
for ((scale=2;scale<=4;scale++))
do
echo "#####current date#####">>output_log.txt
date >>output_log.txt
for ((set=1;set<=5;set++))
do
ori_path=$default_path$set
inter_path=$path$x$scale$interp$t$set$x$scale
block_size=50
reconstrucion_size=400
count=(reconstruction_size/block_size)-1
for ((i=0;i<=3;i++))
do
for ((j=0;j<=3;j++))
do
for ((k=0;k<=3;k++))
do
python generate_subblocks.py --countd $i --counth $j --countw $k --interpath $inter_path --model $model_path >>output_log.txt
done
done
done
python restore_complete_3Dimage.py --oripath $ori_path --interpath $inter_path --sub_name $model$scale$set  >>output_log.txt
done
echo "#####end date#####">>output_log.txt
date >>output_log.txt
done
echo "===================>model end<================" >>output_log.txt
done
