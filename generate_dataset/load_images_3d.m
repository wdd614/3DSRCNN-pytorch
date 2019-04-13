function imgs= load_images_3d(paths)

for i = 1:numel(paths)
    X = imread(paths{i});
    if size(X, 3) == 3 % we extract our features from Y channel
%         X = rgb2ycbcr(X);        
        X = rgb2gray(X);                
        X = X(:, :, 1);
    end
    X = im2double(X); % to reduce memory usage
    imgs(:,:,i)= X;
end