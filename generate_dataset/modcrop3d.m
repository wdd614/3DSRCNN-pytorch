function imgs = modcrop3d(imgs, modulo)
sz=size(imgs);
sz=sz-mod(sz,modulo);
imgs=imgs(1:sz(1),1:sz(2),1:sz(3));
end