function imgs_L = resize3d(imgs, scale, method, verbose)

if nargin < 4
    verbose = 0;
end

h = [];
if verbose
    fprintf('Scaling %d images by %.2f (%s) ', numel(imgs), scale, method);
end

levels = size(imgs);
levels_x=levels(1);
levels_y=levels(2);
levels_z=levels(3);

level_L=1;
for i=1:scale:levels_z
   % h = progress(h, i/numel(imgs), verbose);
   x_image=imgs(:,:,i);
   x_image_H=reshape(x_image,levels_x,levels_y);%这个到底是干嘛的啊？对数组进行重组（矩阵的变形），但元素的个数不变，矩阵大小为levels_x行levels_xy列。这个不是有问题么
    x_image_L= imresize(x_image_H, 1/scale, method);
    imgs_L(:,:,level_L) = x_image_L;
    level_L= level_L+1;
end
if verbose
    fprintf('\n');
end
