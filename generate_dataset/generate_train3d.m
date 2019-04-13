clear;close all;
%% settings
folder = '3dtrain';
savepath = '3dtrain.h5';
size_input = 33;
size_label = 21;2
scale = 2;
stride = 10;

%% initialization
data = zeros(size_input, size_input, size_input, 1);
label = zeros(size_label, size_label, size_label, 1);
padding = abs(size_input - size_label)/2;
count = 0;

%% generate data
% filepaths = dir(fullfile(folder,'*.bmp'));
%     
%     image = imread(fullfile(folder,filepaths(i).name));
%     if size(image,3)==3
%         image=rgb2gray(image);
%         image=im2double(image(:,:,1));
%     end

    image_3d=load_images_3d(glob('3dtrain','*.bmp'));
    im_label = modcrop3d(image_3d, scale);%��ͼ����scale��������
%     [hei, wid, dep] = size(im_label);
%     im_input = imresize(imresize(im_label,1/scale,'bicubic'),[hei,wid], 'bicubic');
%    im_input = interp3(resize3d(image_3d,scale,'bicubic'), scale-1, 'bicubic');
  
   [hei,wid,dep] = size(im_label);
 ori=((1+scale)/2);
[xd,yd,zd] = meshgrid(ori:f:1-ori+wid,ori:f:1-ori+hei,ori:f:1-ori+dep);
[xi,yi,zi] = meshgrid(1:wid,1:hei,1:dep);
 downimage= resize3d(im_label,scale,'bicubic');
  im_input=interp3(xd,yd,zd,downimage,xi,yi,zi, 'cubic');
for i=1:floor(scale/2)
    im_input(:,:,i) =  im_input(:,:,floor(f/2)+1);
   im_input(:,i,:) =  im_input(:,floor(f/2)+1,:);
   im_input(i,:,:) =  im_input(floor(f/2)+1,:,:);
end
for i=1:floor(f/2)
     im_input(:,:,s(3)-i+1) = im_input(:,:,s(3)-floor(f/2));
   im_input(:,s(2)-i+1,:) =  im_input(:,s(2)-floor(f/2),:);
  im_input(s(1)-i+1,:,:) =  im_input(s(1)-floor(f/2),:,:);
end
  
   
    for x = 1 : stride : hei-size_input+1
        for y = 1 :stride : wid-size_input+1
            for z=1 :stride : dep-size_input+1
            
            subim_input = im_input(x : x+size_input-1, y : y+size_input-1,z:z+size_input-1);
            subim_label = im_label(x+padding : x+padding+size_label-1, y+padding : y+padding+size_label-1,z+padding : z+padding+size_label-1);

            count=count+1;
            data(:, :, :, count) = subim_input;
            label(:, :, :, count) = subim_label;
            end
        end
    end


order = randperm(count);
data = data(:, :, :, order);
label = label(:, :, :, order); 

%% writing to HDF5
chunksz = 128;
created_flag = false;
totalct = 0;

for batchno = 1:floor(count/chunksz)
    last_read=(batchno-1)*chunksz;
    batchdata = data(:,:,:,last_read+1:last_read+chunksz); 
    batchlabs = label(:,:,:,last_read+1:last_read+chunksz);

    startloc = struct('dat',[1,1,1,totalct+1], 'lab', [1,1,1,totalct+1]);
    curr_dat_sz = store2hdf5(savepath, batchdata, batchlabs, ~created_flag, startloc, chunksz); 
    created_flag = true;
    totalct = curr_dat_sz(end);
end
h5disp(savepath);
