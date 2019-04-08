CUDA_VISIBLE_DEVICES=1
saved_prefix='low'
default_path='test/testset/test'
x='x'
interp='interp/'
path='test/'
t='test'
mp='model_12layers_25input_3kernel_multi/model_epoch_'
pkl='.pkl'
format='bmp'

for ((model=20;model<=40;model++))
do
model_path=$mp$model$pkl
echo "===================>model start<================ " >> 'output_test.txt'
echo $model_path >>'output_test.txt'
for ((scale=2;scale<=4;scale++))
do
echo "#####current date#####">>'output_test.txt'
date >>'output_test.txt'
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

python generate_subblocks.py --countd $i --counth $j --countw $k --interpath $inter_path --model $model_path --format $format
done
done
done
python restore_complete_3Dimage.py --oripath $ori_path --interpath $inter_path --memo $saved_prefix >>'output_test.txt'
done
echo "#####end date#####">>'output_test.txt'
done
echo "===================>model end<================" >>'output_test.txt'
done
